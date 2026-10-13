<p align="center">
  <img src="assets/colossal-300k.png" alt="A dish of about 300,000 evolving cells, coloured by family" width="620">
</p>

<h1 align="center">origin</h1>
<p align="center"><b>a primordial dish</b></p>

<p align="center">
  <a href="https://origindish.net"><b>Try it live at origindish.net</b></a> &nbsp;·&nbsp;
  <a href="https://x.com/OriginDish">@OriginDish on X</a>
</p>

---

Origin is a living dish of simple cells, each with its own tiny brain. They eat, move, divide and die. Nothing they do is scripted. Every behaviour you see, from families clumping into bodies to hunters chasing prey, comes from the rules plus evolution.

The picture above is about 300,000 cells running in real time on a single graphics card. Each colour is a family.

<p align="center">
  <img src="assets/demo.gif" alt="Cells drifting, feeding and forming bodies in the dish" width="520">
</p>

## What lives in the dish

- **Grazers** eat food that grows in the jelly. Each one is born with a small neural network brain, copied with small changes from its parent.
- **Hunters** are the hollow rings. They eat grazers.
- **Bodies** form when grazers from the same family hold on to each other. Bodies are harder to eat and can fight back.
- **Waste** builds up where cells live and slowly poisons them, so crowding has a cost.
- **Seasons** change how fast food grows back, so good times and hard times come in waves.

<p align="center">
  <img src="assets/closeup.png" alt="Close-up of cell families clumped into bodies, with a hunter nearby" width="520">
</p>

## Two ways to run it

**In your browser** at [origindish.net](https://origindish.net). Every visitor gets their own world, computed on their own device. Small worlds run on the processor. Large and Huge run on the graphics card.

**As a desktop app** (Windows, private build). It is the same simulation built to push the graphics card as far as it goes: dishes up to about a million cells, drawn directly from the card's memory. On an RTX 3090 it holds 300,000+ living cells at a smooth 60 frames per second.

<p align="center">
  <img src="assets/app.png" alt="The desktop app: controls on the left, the dish on the right" width="820">
</p>

## What I have found so far

- **Bodies need a reason to exist.** Cells only form big bodies because staying next to family earns a small food bonus. Turn the bonus off and almost no bodies form.
- **I tried replacing that bonus with something more realistic.** In "outside digestion", cells leak part of their food to the neighbours they hold on to. Bodies still formed and waste deaths dropped sharply, but the dish held far fewer cells, so I kept the original rules.
- **Bigger bodies sit in their own waste.** It is one of the biggest killers in the dish, and an open question whether cells will evolve to move away from it.
- **Each world is different.** Every new world starts from a random seed. Telling real evolution apart from luck means running the same world many times, which is next on the list.

## Built with

- Plain JavaScript in a single page, with the simulation in a background worker
- WebGPU compute shaders for the big worlds, and WebGPU drawing in the desktop app
- Electron for the Windows desktop build

## Credit

Origin started from the cellular automata and neural-network particle work of **Fraser Price** ([@fraserpricee](https://x.com/fraserpricee)). His project was the spark; Origin takes the idea to a much larger scale.

## How it works

Every step, each cell:

1. **Senses** its surroundings: food here and ahead, waste, nearby family and strangers, calls from neighbours, its own energy and a little memory. That is 23 inputs.
2. **Thinks** with a tiny neural network (up to 16 hidden neurons, 9 outputs).
3. **Acts**: swims, turns, brakes, holds on to neighbours, bites, shares, calls out, and remembers.
4. **Pays** for living, moving and thinking, and gains energy from what it eats. Cells next to family get a small food bonus, which is what makes bodies worth forming.
5. **Divides** when it has enough energy. The daughter gets a copy of the brain with small random changes. That is the only source of new behaviour: nothing is designed, it is selected.

Food grows in drifting patches that rise and fall with the seasons. Waste builds up where cells live and slowly poisons them. Hunters eat grazers, but bodies are harder to eat and can sting back.

## The code

```
src/page.html    the page: layout, controls, drawing, story and charts
src/core.js      the simulation on the processor (Small worlds)
src/gpucore.js   the same rules as WebGPU compute shaders (Large and Huge worlds)
src/chron.js     the chronicle: notices records, families and events as they happen
src/worker.js    runs a world in a background worker and sends frames to the page
build.py         joins it all into one file: build/index.html
```

## Run it yourself

- **Easiest:** download `build/index.html` and open it in Chrome or Edge. It is a single file and runs offline. Large and Huge worlds need a browser with WebGPU.
- **After changing the code:** run `python3 build.py`, then open `build/index.html` again.

## Not included

The Windows desktop app (the build that pushes past 300,000 cells) is not part of this repository.

## Licence

The code is released under the [MIT licence](LICENSE): use it, change it and share it, as long as the copyright notice comes along. The name origin, the logo and origindish.net are not part of that licence.
