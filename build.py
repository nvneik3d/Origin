"""Builds the single-file web page: python3 build.py  ->  build/index.html

The page is one HTML file. The simulation code (core.js, gpucore.js, chron.js, worker.js)
is embedded into it and started inside a Web Worker, so the finished page needs no server.
"""
import os

here = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(here, 'src')
read = lambda name: open(os.path.join(src, name), encoding='utf-8').read()

engine = '\n'.join(read(f) for f in ['core.js', 'gpucore.js', 'chron.js', 'worker.js'])
page = read('page.html').replace('/*CORE*/', engine)
head = '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n'

os.makedirs(os.path.join(here, 'build'), exist_ok=True)
with open(os.path.join(here, 'build', 'index.html'), 'w', encoding='utf-8') as f:
    f.write(head + page)
print('wrote build/index.html')
