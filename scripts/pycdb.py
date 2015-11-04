'''Copyright (c) 2009-2015 David Wilson <dw@botanicus.net>

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

From https://github.com/dw/python-pure-cdb

Modified to remove unnecessary bits (writer class, and reader decoding
wrappers).

Manipulate DJB's Constant Databases. These are 2 level disk-based hash tables
that efficiently handle many keys, while remaining space-efficient.

    http://cr.yp.to/cdb.html
'''

from _struct import Struct
from itertools import chain

def py_djb_hash(s):
    '''Return the value of DJB's hash function for the given 8-bit string.'''
    h = 5381
    for c in s:
        h = (((h << 5) + h) ^ ord(c)) & 0xffffffff
    return h

try:
    from _cdblib import djb_hash
except ImportError:
    djb_hash = py_djb_hash


read_2_le4 = Struct('<LL').unpack
write_2_le4 = Struct('<LL').pack


class Reader(object):
    '''A dictionary-like object for reading a Constant Database accessed
    through a string or string-like sequence, such as mmap.mmap().'''

    def __init__(self, data, hashfn=djb_hash):
        '''Create an instance reading from a sequence and using hashfn to hash
        keys.'''
        if len(data) < 2048:
            raise IOError('CDB too small')

        self.data = data
        self.hashfn = hashfn

        self.index = [read_2_le4(data[i:i+8]) for i in xrange(0, 2048, 8)]
        self.table_start = min(p[0] for p in self.index)
        # Assume load load factor is 0.5 like official CDB.
        self.length = sum(p[1] >> 1 for p in self.index)

    def iteritems(self):
        '''Like dict.iteritems(). Items are returned in insertion order.'''
        pos = 2048
        while pos < self.table_start:
            klen, dlen = read_2_le4(self.data[pos:pos+8])
            pos += 8

            key = self.data[pos:pos+klen]
            pos += klen

            data = self.data[pos:pos+dlen]
            pos += dlen

            yield key, data

    def items(self):
        '''Like dict.items().'''
        return list(self.iteritems())

    def iterkeys(self):
        '''Like dict.iterkeys().'''
        return (p[0] for p in self.iteritems())
    __iter__ = iterkeys

    def itervalues(self):
        '''Like dict.itervalues().'''
        return (p[1] for p in self.iteritems())

    def keys(self):
        '''Like dict.keys().'''
        return [p[0] for p in self.iteritems()]

    def values(self):
        '''Like dict.values().'''
        return [p[1] for p in self.iteritems()]

    def __getitem__(self, key):
        '''Like dict.__getitem__().'''
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def has_key(self, key):
        '''Return True if key exists in the database.'''
        return self.get(key) is not None
    __contains__ = has_key

    def __len__(self):
        '''Return the number of records in the database.'''
        return self.length

    def gets(self, key):
        '''Yield values for key in insertion order.'''
        # Truncate to 32 bits and remove sign.
        h = self.hashfn(key) & 0xffffffff
        start, nslots = self.index[h & 0xff]

        if nslots:
            end = start + (nslots << 3)
            slot_off = start + (((h >> 8) % nslots) << 3)

            for pos in chain(xrange(slot_off, end, 8),
                             xrange(start, slot_off, 8)):
                rec_h, rec_pos = read_2_le4(self.data[pos:pos+8])

                if not rec_h:
                    break
                elif rec_h == h:
                    klen, dlen = read_2_le4(self.data[rec_pos:rec_pos+8])
                    rec_pos += 8

                    if self.data[rec_pos:rec_pos+klen] == key:
                        rec_pos += klen
                        yield self.data[rec_pos:rec_pos+dlen]

    def get(self, key, default=None):
        '''Get the first value for key, returning default if missing.'''
        # Avoid exception catch when handling default case; much faster.
        return chain(self.gets(key), (default,)).next()
