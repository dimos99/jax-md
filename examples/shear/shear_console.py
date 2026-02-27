"""Console logging helpers with optional colorama support."""

import os
import sys

try:
  from colorama import Fore
  from colorama import Style
  from colorama import init as _colorama_init
  _HAS_COLORAMA = True
except Exception:  # pragma: no cover
  Fore = None
  Style = None
  _colorama_init = None
  _HAS_COLORAMA = False


def _wants_color() -> bool:
  if os.environ.get('NO_COLOR') is not None:
    return False
  forced = os.environ.get('RPY_SHEAR_COLOR')
  if forced is not None:
    return forced.strip().lower() in ('1', 'true', 'yes', 'on')
  return sys.stdout.isatty()


class Console:
  """Small structured logger for human-readable progress output."""

  def __init__(self):
    self.use_color = bool(_HAS_COLORAMA and _wants_color())
    if _HAS_COLORAMA:
      _colorama_init(autoreset=True, strip=not self.use_color)

  def _prefix(self, tag: str, color: str = '') -> str:
    base = f'[{tag}]'
    if self.use_color and color:
      return f'{color}{base}{Style.RESET_ALL}'
    return base

  def section(self, title: str):
    print(self._prefix('SECTION', Fore.WHITE if Fore else '') + f' {title}')

  def info(self, msg: str):
    print(self._prefix('INFO', Fore.CYAN if Fore else '') + f' {msg}')

  def warn(self, msg: str):
    print(self._prefix('WARN', Fore.YELLOW if Fore else '') + f' {msg}')

  def error(self, msg: str):
    print(self._prefix('ERROR', Fore.RED if Fore else '') + f' {msg}')

  def progress(self, msg: str):
    print(self._prefix('RUN', Fore.MAGENTA if Fore else '') + f' {msg}')

  def success(self, msg: str):
    print(self._prefix('DONE', Fore.GREEN if Fore else '') + f' {msg}')


_CONSOLE = None


def get_console() -> Console:
  global _CONSOLE
  if _CONSOLE is None:
    _CONSOLE = Console()
  return _CONSOLE
