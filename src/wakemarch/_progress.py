"""Small progress display: native notebook HTML or dependency-free terminal text."""
import sys
import time


class ProgressBar:
    def __init__(self):
        self.started = time.monotonic()
        self.last_update = 0.0
        self.completed = 0.0
        self.total = None
        self.display = None
        # IPython is optional and imported only if a notebook kernel is running.
        if 'IPython' in sys.modules:
            from IPython import get_ipython
            if getattr(get_ipython(), 'kernel', None) is not None:
                from IPython.display import HTML, display
                self._html = HTML
                self.display = display(HTML(self._markup('Preparing solver', None)), display_id=True)
        if self.display is None:
            self._text('Preparing solver', None)

    def _markup(self, label, fraction):
        value = '' if fraction is None else f' value="{fraction:.6f}"'
        return (f'<div><strong>WakeMarch</strong> · {label}<br>'
                f'<progress aria-label="WakeMarch solve progress" max="1"{value} '
                'style="width: min(100%, 480px); height: 18px;"></progress></div>')

    def _text(self, label, fraction):
        bar = '' if fraction is None else '[' + '#' * int(24 * fraction) + '-' * (24 - int(24 * fraction)) + '] '
        sys.stderr.write('\r' + f'WakeMarch {bar}{label}'.ljust(110))
        sys.stderr.flush()

    def _render(self, label, fraction):
        if self.display is not None:
            self.display.update(self._html(self._markup(label, fraction)))
        else:
            self._text(label, fraction)

    def __call__(self, completed_D, total_D):
        self.completed, self.total = completed_D, total_D
        now = time.monotonic()
        if now - self.last_update < 0.1 and completed_D < total_D:
            return
        self.last_update = now
        fraction = completed_D / total_D
        self._render(f'Marching {fraction:.0%} · {completed_D:.1f}/{total_D:.1f} D '
                     f'(including buffer) · {now-self.started:.1f}s', fraction)

    def close(self, success):
        fraction = None if self.total is None else self.completed / self.total
        label = 'Complete' if success else 'Stopped before completion'
        self._render(f'{label} · {time.monotonic()-self.started:.1f}s',
                     1.0 if success else fraction)
        if self.display is None:
            sys.stderr.write('\n')
            sys.stderr.flush()
