"""
termcolor: https://pypi.org/project/termcolor/
examples:
- https://pypi.org/project/termcolor/
- https://www.codegrepper.com/code-examples/python/python+color+text+on+windows
- https://www.programcreek.com/python/example/78943/termcolor.colored
properties:
- text colors:
  - grey
  - red
  - green
  - yellow
  - blue
  - magenta
  - cyan
  - white
- text highlights:
  - on_grey
  - on_red
  - on_green
  - on_yellow
  - on_blue
  - on_magenta
  - on_cyan
  - on_white
- attributes:
  - bold
  - dark
  - underline
  - blink
  - reverse
  - concealed
"""
# pylint: disable=wrong-import-position

# this is important, definitely in windows
import os

os.system("color")
# (end) this is important, definitely in windows

import sys
from typing import Callable
from termcolor import colored, cprint

# my examples
print(colored(text="HELLO NO BOLD", color="green", on_color=None))
print(colored(text="HELLO BOLD", color="green", on_color=None, attrs=["bold"]))
print(colored(text="HI", color="red", on_color="on_white", attrs=["bold"]))
print(colored(text="HI", color="red", on_color=None, attrs=["concealed"]))
print("\n")

# examples from https://pypi.org/project/termcolor/
text = colored('Hello, World!', 'red', attrs=['reverse', 'blink'])
print(text)
cprint('Hello, World!', 'green', 'on_red')

print_red_on_cyan: Callable[[str], None] = lambda x: cprint(x, 'red', 'on_cyan')
print_red_on_cyan('Hello, World!')
print_red_on_cyan('Hello, Universe!')

for i in range(10):
    cprint(str(i), 'magenta', end=' ')

cprint("Attention!", 'red', attrs=['bold'], file=sys.stderr)
# pylint: enable=wrong-import-position
