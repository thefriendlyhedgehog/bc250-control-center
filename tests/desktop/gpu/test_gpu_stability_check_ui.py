from types import SimpleNamespace

from frontends.desktop.i18n import SUPPORTED_LANGUAGES, tr
from frontends.desktop.pages.gpu_governor import GpuGovernorPage

SOURCES = (
    "Test stability",
    "Run memtest_vulkan for 10 minutes at the governor's clocks and watch the kernel log for GPU faults. Nothing is changed; an unstable GPU can hang the board.",
    "Stability check",
    "Not run",
    "memtest_vulkan with an amdgpu fault watch",
    "Could not open the stability check",
    "Opened the GPU stability check in the console.",
    "{minutes} min at {clock} MHz, edge {edge} °C · {when}",
)


class Controller:
    def __init__(self, result=None):
        self.result = result
        self.opened = []

    def ultima_prueba_estabilidad_gpu(self):
        return self.result

    def abrir_prueba_estabilidad_gpu(self, minutos):
        self.opened.append(minutos)
        return "terminal"


def _view(page):
    return page._redesigned_gpu_view


def test_the_stability_panel_sits_in_the_visible_gpu_view(qtbot):
    page = GpuGovernorPage(Controller())
    qtbot.addWidget(page)
    view = _view(page)
    assert view.stability_button.text() == "Test stability"
    assert "Nothing is changed" in view.stability_button.toolTip()
    assert view._stability_outcome.text() == "Not run"
    view.set_oberon_mode(True)
    assert not view.stability_panel.isHidden()


def test_the_panel_shows_the_newest_check(qtbot):
    controller = Controller()
    page = GpuGovernorPage(controller)
    qtbot.addWidget(page)
    controller.result = SimpleNamespace(
        at=1_760_000_000.0, outcome="PASS", seconds=600, average_clock_mhz=1998, max_edge_c=84,
    )
    page._sync_stability_line()
    view = _view(page)
    assert view._stability_outcome.text() == "PASS"
    assert view._stability_detail.text().startswith("10 min at 1998 MHz, edge 84 °C · ")


def test_a_broken_results_file_reads_as_not_run(qtbot):
    class Broken(Controller):
        def ultima_prueba_estabilidad_gpu(self):
            raise ValueError("bad json")

    page = GpuGovernorPage(Broken())
    qtbot.addWidget(page)
    page._sync_stability_line()
    assert _view(page)._stability_outcome.text() == "Not run"


def test_the_button_opens_a_ten_minute_check_in_the_console(qtbot, monkeypatch):
    controller = Controller()
    page = GpuGovernorPage(controller)
    qtbot.addWidget(page)
    calls = []

    def run(operation, on_success, error_title, **kwargs):
        calls.append((error_title, kwargs))
        on_success(operation())
        return True

    monkeypatch.setattr(page, "_run_backend_action", run)
    _view(page).stability_button.click()
    assert controller.opened == [10]
    assert calls == [("Could not open the stability check", {"refresh": False})]
    assert page.last_operation_line.detail.text() == "Opened the GPU stability check in the console."


def test_the_stability_strings_are_translated_in_every_language():
    for language in SUPPORTED_LANGUAGES - {"en"}:
        missing = [source for source in SOURCES if tr(source, language) == source]
        assert missing == [], language
