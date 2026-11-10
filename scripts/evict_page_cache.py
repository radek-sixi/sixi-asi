#!/usr/bin/env python3
"""Drop the page cache of the given files (posix_fadvise DONTNEED, no root needed). On Thor's unified memory a
large page cache left by earlier model loads can starve the next process's device allocations (the allocator spins
in the kernel instead of reclaiming), so experiment drivers evict model files before starting a server."""
import os, sys, glob
n = 0
for pat in sys.argv[1:]:
    for p in glob.glob(os.path.expanduser(pat), recursive=True):
        p = os.path.realpath(p)
        if not os.path.isfile(p):
            continue
        fd = os.open(p, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            n += 1
        finally:
            os.close(fd)
print(f"evicted page cache of {n} files")
