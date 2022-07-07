import io
from collections import namedtuple

FileRange = namedtuple("FileRange", ["file", "start", "end"])

SEEK_SET = 0
SEEK_CUR = 1
SEEK_END = 2


def _file_size(file):
    start = file.tell()
    size = file.seek(0, 2)
    file.seek(start)
    return size


def _iter_file_ranges(files):
    start = 0
    end = 0
    for file in files:
        size = _file_size(file)
        start = end
        end = start + size
        yield FileRange(file, start, end)


class SplitFile(io.RawIOBase):
    def __init__(self, files):
        self._offset = 0
        self._ranges = list(_iter_file_ranges(files))
    
    @classmethod
    def from_paths(cls, paths, mode="rb"):
        files = [open(path, "rb") for path in paths]
        return cls(files=files)

    @property
    def _end(self):
        try:
            return self._ranges[-1].end
        except IndexError:
            return 0

    def readable(self):
        return True

    def writable(self):
        return False

    def seekable(self):
        return True

    def close(self):
        for range in self._ranges:
            range.file.close()

    def seek(self, offset, whence=0):
        if whence == SEEK_CUR:
            offset = self.tell() + offset
        elif whence == SEEK_END:
            offset = self._end + offset
        
        if offset < 0:
            raise ValueError("negative seek position {}".format(offset))

        self._offset = offset
        
        return self._offset

    def tell(self):
        return self._offset

    def _range_for_offset(self, offset):
        for range in self._ranges:
            if offset >= range.start and offset < range.end:
                return range

    def read(self, size=None):
        # Read from each file starting from tell() until size is exhausted
        result = bytearray()

        count = self._readinto(result, size=size)

        return bytes(result)

    def readinto(self, result):
        return self._readinto(result)

    def _readinto(self, result, size=None):
        if size is None:
            size = self._end

        bytes_read = 0
        read_end = min(self._end, self._offset + size)

        while self._offset < read_end:
            range = self._range_for_offset(self._offset)
            range.file.seek(self._offset - range.start)

            remaining_in_range = min(range.end - self._offset, read_end - self._offset)

            result.extend(range.file.read(remaining_in_range))
            self._offset += remaining_in_range
            bytes_read += remaining_in_range

        return bytes_read
