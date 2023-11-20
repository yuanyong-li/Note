## Vim

## Peek

### Ch02. Buffers_Windows_Tabs

```vim
# Buffers
vim file1.py file2.py
:bnext
:bprevious
:buffer + filename
:buffer + n
--------------------------------
# Windows
Ctrl-W H    Moves the cursor to the left window
Ctrl-W J    Moves the cursor to the window below
Ctrl-W K    Moves the cursor to the window upper
Ctrl-W L    Moves the cursor to the right window

Ctrl-W V    Opens a new vertical split
Ctrl-W S    Opens a new horizontal split
Ctrl-W C    Closes a window
Ctrl-W O    Makes the current window the only one on screen and closes other windows

:vsplit filename    Split window vertically
:split filename     Split window horizontally
:new filename       Create new window
--------------------------------
# Tabs
:tabnew file.txt    Open file.txt in a new tab
:tabclose           Close the current tab
:tabnext            Go to next tab
:tabprevious        Go to previous tab
:tablast            Go to last tab
:tabfirst           Go to first tab

vim -p file1.sj file2.js file3.js
```

### Ch03. Searching Files

```vim
--------------------------------
:set path?
:set path+=app/controllers/
# or add `set path+=app/controllers/` into ~/.vimrc

:edit file.txt
:edit **/*.md<tab>
:find file.txt
--------------------------------
--------------------------------
--------------------------------
--------------------------------
```



## Ch01. Starting Vim

If you want to open the file `hello.txt` and immediately execute a Vim command, you can pass to the `vim` command the `+{cmd}` option.

```bash
vim +%s/pancake/bagel/g +%s/bagel/egg/g +%s/egg/donut/g hello.txt
```

## Ch02. Buffers_Windows_Tabs

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

## Ch03. Searching Files

**edit**、**find**、*****、******的用法

```
:edit file.txt
:find file.txt
```

 `:find` finds file in `path`, `:edit` doesn't.

```
:set path?
```

By default, yours probably look like this:

```
path=.,/usr/include,,
```

- `.` means to search in the directory of the currently opened file.
- `,` means to search in the current directory.
- `/usr/include` is the typical directory for C libraries header files.

```
:set path+=app/controllers/
```

or you can add `set path+=app/controllers/` into `~/.vimrc`

---

You can use `**` to search recursively. If you want to look for all `*.md` files in your project, but you are not sure in which directories, you can do this:

```
:edit **/*.md<Tab>
```

### Searching in Files with Grep

If you need to find in files (find phrases in files), you can use grep. Vim has two ways of doing that:

- Internal grep (`:vim`. Yes, it is spelled `:vim`. It is short for `:vimgrep`).
- External grep (`:grep`).

Let's go through internal grep first. `:vim` has the following syntax:

```
:vim /pattern/ file
```

- `/pattern/` is a regex pattern of your search term.
- `file` is the file argument. You can pass multiple arguments. Vim will search for the pattern inside the file argument. Similar to `:find`, you can pass it `*` and `**` wildcards.

```
:vim /breakfast/ app/controllers/**/*.rb
```

After running that, you will be redirected to the first result. Vim's `vim` search command uses `quickfix` operation. To see all search results, run `:copen`. This opens a `quickfix` window. Here are some useful **quickfix** commands to get you productive immediately:

```
:copen        Open the quickfix window
:cclose       Close the quickfix window
:cnext        Go to the next error
:cprevious    Go to the previous error
:colder       Go to the older error list
:cnewer       Go to the newer error list
```

### Netrw

`netrw` is Vim's built-in file explorer. It is useful to see a project's hierarchy. To run `netrw`, you need these two settings in your `.vimrc`:

```
set nocp
filetype plugin on
```

```
:edit .
```

There are other ways to launch `netrw` window without passing a directory:

```
:Explore     Starts netrw on current file
:Sexplore    No kidding. Starts netrw on split top half of the screen
:Vexplore    Starts netrw on split left half of the screen
```

```
%    Create a new file
d    Create a new directory
R    Rename a file or directory
D    Delete a file or directory
```

### Fzf & RipGrep

First, make sure you have [fzf](https://github.com/junegunn/fzf) and [ripgrep](https://github.com/BurntSushi/ripgrep) downloaded. Follow the instruction on their github repo. The commands `fzf` and `rg` should now be available after successful installs.

Fzf does not use ripgrep by default, so we need to tell fzf to use ripgrep by defining a `FZF_DEFAULT_COMMAND` variable. In my `.zshrc` (`.bashrc` if you use bash), I have these:

```
if type rg &> /dev/null; then
  export FZF_DEFAULT_COMMAND='rg --files'
  export FZF_DEFAULT_OPTS='-m'
fi
```

Pay attention to `-m` in `FZF_DEFAULT_OPTS`. This option allows us to make multiple selections with `<Tab>` or `<Shift-Tab>`. You don't need this line to make fzf work with Vim, but I think it is a useful option to have. It will come in handy when you want to perform search and replace in multiple files which I'll cover in just a little. The fzf command accepts many more options, but I won't cover them here. To learn more, check out [fzf's repo](https://github.com/junegunn/fzf#usage) or `man fzf`. At minimum you should have `export FZF_DEFAULT_COMMAND='rg'`.

After installing fzf and ripgrep, let's set up the fzf plugin. I am using [vim-plug](https://github.com/junegunn/vim-plug) plugin manager in this example, but you can use any plugin managers.

Add these inside your `.vimrc` plugins. You need to use [fzf.vim](https://github.com/junegunn/fzf.vim) plugin (created by the same fzf author).

```
call plug#begin()
Plug 'junegunn/fzf.vim'
Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
call plug#end()
```

After adding these lines, you will need to open `vim` and run `:PlugInstall`. It will install all plugins that are defined in your `vimrc` file and are not installed. In our case, it will install `fzf.vim` and `fzf`. 

For more info about this plugin, you can check out [fzf.vim repo](https://github.com/junegunn/fzf/blob/master/README-VIM.md).

### Finding Files

To search for files inside Vim using fzf.vim plugin, you can use the `:Files` method. Run `:Files` from Vim and you will be prompted with fzf search prompt.

### Finding in Files

To search inside files, you can use the `:Rg` command.

### Replacing Grep With Rg

Add this in your vimrc:

```
set grepprg=rg\ --vimgrep\ --smart-case\ --follow
```

## Search and Replace in Multiple Files

Modern text editors like VSCode makes it very easy to search and replace a string across multiple files. In this section, I will show you two different methods to easily do that in Vim.

The first method is to replace *all* matching phrases in your project. You will need to use `:grep`. If you want to replace all instances of "pizza" with "donut", here's what you do:

```
:grep "pizza"
:cfdo %s/pizza/donut/g | update
```

Let's break down the commands:

1. `:grep pizza` uses ripgrep to search for all instances of "pizza" (by the way, this would still work even if you didn't reassign `grepprg` to use ripgrep. You would have to do `:grep "pizza" . -R` instead of `:grep "pizza"`).
2. `:cfdo` executes any command you pass to all files in your quickfix list. In this case, your command is the substitution command `%s/pizza/donut/g`. The pipe (`|`) is a chain operator. The `update` command saves each file after substitution. I will cover the substitute command in more depth in a later chapter.

The second method is to search and replace in selected files. With this method, you can manually choose which files you want to perform select-and-replace on. Here is what you do:

1. Clear your buffers first. It is imperative that your buffer list contains only the files you want to apply the replace on. You can either restart Vim or run `:%bd | e#` command (`%bd` deletes all the buffers and `e#` opens the file you were just on).
2. Run `:Files`.
3. Select all files you want to perform search-and-replace on. To select multiple files, use `<Tab>` / `<Shift-Tab>`. This is only possible if you have the multiple flag (`-m`) in `FZF_DEFAULT_OPTS`.
4. Run `:bufdo %s/pizza/donut/g | update`. The command `:bufdo %s/pizza/donut/g | update` looks similar to the earlier `:cfdo %s/pizza/donut/g | update` command. The difference is instead of substituting all quickfix entries (`:cfdo`), you are substituting all buffer entries (`:bufdo`).

## Ch04. Vim Grammer

### Nouns (Motions)

```
h    Left
j    Down
k    Up
l    Right
w    Move forward to the beginning of the next word
}    Jump to the next paragraph
$    Go to the end of the line
G	 Go to the line(2G, 101G)
```

### Verbs (Operators)

`h: operator`

```
y    Yank text (copy)
d    Delete text and save to register
c    Delete text, save to register, and start insert mode
```

As a side note, linewise operations (operations affecting the entire line) are common operations in text editing. In general, by typing an operator command twice, Vim performs a linewise operation for that action. For example, `dd`, `yy`, and `cc` perform **deletion**, **yank**, and **change** on the entire line. Try this with other operators!

### More Nouns (Text Objects)

```
i + object    Inner text object
a + object    Outer text object
```

Inner text object selects the object inside *without* the white space or the surrounding objects. Outer text object selects the object inside *including* the white space or the surrounding objects. Generally, an outer text object always selects more text than an inner text object.

`(hello Vim)`:

- To delete the text inside the parentheses without deleting the parentheses: `di(`.
- To delete the parentheses and the text inside: `da(`.

```javascript
const hello = function() {
  console.log("Hello Vim");
  return true;
}
```

- To delete the entire "Hello Vim": `di(`.
- To delete the content of function (surrounded by `{}`): `di{`.
- To **delete the "Hello" string**: `diw`.

```
w         A word
p         A paragraph
s         A sentence
( or )    A pair of ( )
{ or }    A pair of { }
[ or ]    A pair of [ ]
< or >    A pair of < >
t         XML tags
"         A pair of " "
'         A Pair of ' '
`         A pair of ` `
```

## Ch05. Moving in a File

