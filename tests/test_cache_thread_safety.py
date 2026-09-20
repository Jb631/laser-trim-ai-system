"""The two caches the ingest workers share must survive being shared.

The ingest pool is a ThreadPoolExecutor. `_read_once`'s byte cache (8 entries) and
`hashing`'s hash cache are module-level dicts that every worker reads, reorders and
evicts from. Each of those was a read-then-write with nothing holding the two halves
together: `get()` then `move_to_end()`, `in` then `[]`, `list(keys)` then `del`. A
worker evicting between another worker's two halves raised KeyError, which
`Processor.process_file` records as an ERROR row -- a GOOD file lost, silently, in an
unattended 151,000-file rebuild. Found by code review 2026-09-20; reproduced with the
loop below (325 exceptions in 6 s) before the lock went in.
"""
import collections
import sys
import threading
import time


def test_the_shared_caches_survive_eight_workers(tmp_path, monkeypatch):
    from laser_trim_analyzer.core import parser
    from laser_trim_analyzer.utils import hashing

    files = []
    for i in range(64):
        p = tmp_path / f"f{i}.bin"
        p.write_bytes(bytes([i]) * 64)
        files.append(p)
    # Constant eviction in the hash cache too (the byte cache already holds only 8).
    monkeypatch.setattr(hashing, "_cache_max_size", 16)
    hashing.clear_hash_cache()

    errors = collections.Counter()
    served_wrong = []
    stop = time.time() + 2.0
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)                 # switch threads between bytecodes, not every 5 ms

    def work(seed):
        i = seed
        while time.time() < stop:
            p = files[i % 64]
            i += 7
            try:
                data = parser._read_once(p)
                if data != bytes([int(p.stem[1:])]) * 64:
                    served_wrong.append(p.name)             # a lock must never buy speed with a wrong answer
                hashing.hash_bytes_for(p, data)
                hashing.calculate_file_hash(p)
                if i % 5 == 0:
                    parser.drop_cached_bytes(p)
            except Exception as exc:                        # noqa: BLE001 -- the point is to see ALL of them
                errors[f"{type(exc).__name__}: {exc}"[:80]] += 1

    try:
        threads = [threading.Thread(target=work, args=(s,)) for s in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
    finally:
        sys.setswitchinterval(old)
        hashing.clear_hash_cache()

    assert not any(t.is_alive() for t in threads), "a worker never finished -- deadlock"
    assert dict(errors) == {}
    assert served_wrong == []


def test_the_slow_read_happens_outside_the_lock(tmp_path, monkeypatch):
    """The lock guards dictionary bookkeeping only. If it were held across read_bytes(),
    four workers on the share would queue behind one 600 ms network read each -- the
    single-read fix would have made the rebuild SLOWER. Two threads must be inside
    read_bytes() at the same moment."""
    from pathlib import Path
    from laser_trim_analyzer.core import parser

    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"a" * 8)
    b.write_bytes(b"b" * 8)
    inside = threading.Barrier(2, timeout=10)
    real = Path.read_bytes

    def slow_read(self):
        inside.wait()                           # only passes once BOTH threads are in here together
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", slow_read)
    out = {}
    threads = [threading.Thread(target=lambda p=p: out.__setitem__(p.name, parser._read_once(p))) for p in (a, b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15)
    assert not any(t.is_alive() for t in threads)
    assert out == {"a.bin": b"a" * 8, "b.bin": b"b" * 8}
