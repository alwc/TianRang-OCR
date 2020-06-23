#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/28 17:00


import os
from os import path
from datetime import datetime


def get_day():
    now = datetime.today()
    return now.day


class FileRecord:
    def __init__(self, dir_path, prefix, flush=False):
        if not path.exists(dir_path):
            os.makedirs(dir_path)

        self.dir_path = dir_path
        self.prefix = prefix
        self.file = self._open_file()
        self.flush = flush

    def write(self, s):
        now = datetime.today()
        if self.day != now.day:
            self.file.flush()
            self.file.close()
            self.file = self._open_file()

        self.file.write(s + "\n")

        if self.flush:
            self.file.flush()

    def _open_file(self):
        now = datetime.today()
        self.day = now.day
        filename = "{prefix}_{year:04d}-{month:02d}-{day:02d}".format(
            prefix=self.prefix, year=now.year, month=now.month, day=now.day)
        file_path = path.join(self.dir_path, filename)

        return open(file_path, "a+")
