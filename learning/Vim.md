## Vim

## Ch01. Starting Vim

If you want to open the file `hello.txt` and immediately execute a Vim command, you can pass to the `vim` command the `+{cmd}` option.

```bash
vim +%s/pancake/bagel/g +%s/bagel/egg/g +%s/egg/donut/g hello.txt
```

## Ch02. 

make sure having the `set hidden` option in vimrc.

`vim ~/.vimrc`

inside it, add:

```
set hidden
```

Save it, then source it (run `:source %` from inside the vimrc).

### Buffers

```bash
vim file1.py file2.py
```

There are several ways you can traverse buffers:

- `:bnext` to go to the next buffer (`:bprevious` to go to the previous buffer).
- `:buffer` + filename. Vim can autocomplete filename with `<Tab>`.
- `:buffer` + `n`, where `n` is the buffer number. For example, typing `:buffer 2` will take you to buffer #2.
- Jump to the older position in the jump list with `Ctrl-O` and to the newer position with `Ctrl-I`. These are not buffer specific methods, but they can be used to jump between different buffers. I will explain jumps in further details in Chapter 5.
- Go to the previously edited buffer with `Ctrl-^`.

if have multiple buffers opened, you can close all of them with quit-all:

```
:qall
```

```
:wqa!
```

### Windows

```
vim file1.py
```

```
:split file1.py
```

If you want to navigate between windows, use these shortcuts:

```
Ctrl-W H    Moves the cursor to the left window
Ctrl-W J    Moves the cursor to the window below
Ctrl-W K    Moves the cursor to the window upper
Ctrl-W L    Moves the cursor to the right window

Ctrl-W V    Opens a new vertical split
Ctrl-W S    Opens a new horizontal split
Ctrl-W C    Closes a window
Ctrl-W O    Makes the current window the only one on screen and closes other windows
```

In the window, use `buffer file2.py` to display other buffer.

And here is a list of useful window command-line commands:

```
:vsplit filename    Split window vertically
:split filename     Split window horizontally
:new filename       Create new window
```

### Tabs

```bash
vim file1.py
```

open `file2.py` in a new tab

```
:tabnew file2.py
```

Below is a list of useful tab navigations:

```
:tabnew file.txt    Open file.txt in a new tab
:tabclose           Close the current tab
:tabnext            Go to next tab
:tabprevious        Go to previous tab
:tablast            Go to last tab
:tabfirst           Go to first tab
```

You can also run `gt` to go to next tab page (you can go to previous tab with `gT`). You can pass count as argument to `gt`, where count is tab number. To go to the third tab, do `3gt`.

To start Vim with multiple tabs, you can do this from the terminal:

```bash
vim -p file1.js file2.js file3.js
```

### The Way to Use Buffers, Windows, and Tabs

Everyone has a different workflow, here is mine for example:

- First, I use buffers to store all the required files for the current task. Vim can handle many opened buffers before it starts slowing down. Plus having many buffers opened won't crowd my screen. I am only seeing one buffer (assuming I only have one window) at any time, allowing me to focus on one screen. When I need to go somewhere, I can quickly fly to any open buffer anytime.
- I use multiple windows to view multiple buffers at once, usually when diffing files, reading docs, or following a code flow. I try to keep the number of windows opened to no more than three because my screen will get crowded (I use a small laptop). When I am done, I close any extra windows. Fewer windows means less distractions.
- Instead of tabs, I use [tmux](https://github.com/tmux/tmux/wiki) windows. I usually use multiple tmux windows at once. For example, one tmux window for client-side codes and another for backend codes.