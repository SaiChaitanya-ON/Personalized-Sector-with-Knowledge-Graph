import mock
import pytest

from onai.ml.pdf.extraction import LTChar, LTTextLine, content_chars, content_lines


@pytest.fixture
def mock_miner():
    with mock.patch(
        "onai.ml.pdf.extraction.PDFParser", autospec=True
    ) as mparser, mock.patch(
        "onai.ml.pdf.extraction.PDFDocument", autospec=True
    ) as mdoc, mock.patch(
        "onai.ml.pdf.extraction.PDFResourceManager", autospec=True
    ) as mrm, mock.patch(
        "onai.ml.pdf.extraction.PDFLayoutAnalyzer", autospec=True
    ) as mla, mock.patch(
        "onai.ml.pdf.extraction.PDFPageInterpreter", autospec=True
    ) as mint, mock.patch(
        "onai.ml.pdf.extraction.PDFPage", autospec=True
    ) as mpage:
        mla.return_value.cur_item = None
        yield {
            "parser": mparser,
            "doc": mdoc,
            "rm": mrm,
            "device": mla,
            "int": mint,
            "page": mpage,
        }


def dummy_page():
    p = mock.MagicMock()
    p.mediabox = (0.0, 0.0, 1.0, 1.0)
    p.cropbox = (0.0, 0.0, 1.0, 1.0)
    return p


class fakefont:
    fontname = "fakefont"

    def is_vertical(self):
        return False

    def get_descent(self):
        return 0.0


def char(c, x0, x1, y0, y1):
    return LTChar(
        (x1 - x0, 0, 0, y1 - y0, x0, y0),
        fakefont(),
        1,
        1.0,
        0.0,
        c,
        1.0,
        None,
        None,
        None,
    )


def test_contents_path(mock_miner):
    with mock.patch("onai.ml.pdf.extraction._contents", autospec=True) as m_conts:
        with mock.patch(
            "builtins.open", mock.mock_open(read_data=b"this is a test")
        ) as m_open:
            content_lines("test/path")
            assert m_open.call_args == mock.call("test/path", "rb")
            assert m_conts.call_args == mock.call(m_open.return_value, LTTextLine)


def test_contents_file(mock_miner):
    with mock.patch("onai.ml.pdf.extraction._contents", autospec=True) as m_conts:
        with mock.patch(
            "builtins.open", mock.mock_open(read_data=b"this is a test")
        ) as m_open:
            content_lines(m_open.return_value)
            assert m_conts.call_args == mock.call(m_open.return_value, LTTextLine)


def test_contents_file_char(mock_miner):
    with mock.patch("onai.ml.pdf.extraction._contents", autospec=True) as m_conts:
        with mock.patch(
            "builtins.open", mock.mock_open(read_data=b"this is a test")
        ) as m_open:
            content_chars(m_open.return_value)
            assert m_conts.call_args == mock.call(m_open.return_value, LTChar)


def test_pdf_page_extr(mock_miner):
    with mock.patch(
        "builtins.open", mock.mock_open(read_data=b"this is a test")
    ) as m_open:
        content_lines("test/path")
        assert mock_miner["parser"].call_args == mock.call(m_open.return_value)
        assert mock_miner["doc"].call_args == mock.call(
            mock_miner["parser"].return_value
        )
        assert mock_miner["page"].create_pages.call_args == mock.call(
            mock_miner["doc"].return_value
        )


def test_multipage_retrieval(mock_miner):
    with mock.patch(
        "builtins.open", mock.mock_open(read_data=b"this is a test")
    ) as m_open:
        pages = [dummy_page(), dummy_page(), dummy_page()]
        mock_miner["page"].create_pages.return_value = pages

        r = content_lines("test/path")
        assert mock_miner["parser"].call_args == mock.call(m_open.return_value)
        assert mock_miner["int"].return_value.process_page.call_args_list == [
            mock.call(p) for p in pages
        ]
        assert len(r) == len(pages)


def test_page_cont_extraction_line(mock_miner):
    with mock.patch("builtins.open", mock.mock_open(read_data=b"this is a test")):
        mock_miner["page"].create_pages.return_value = [dummy_page()]
        line = LTTextLine(0.3)
        line.bbox = (0.25, 0.25, 0.75, 0.75)
        char1 = char("T", 0.25, 0.5, 0.25, 0.5)
        char2 = char("e", 0.25, 0.5, 0.25, 0.5)
        char3 = char("s", 0.25, 0.5, 0.25, 0.5)
        char4 = char("t", 0.25, 0.5, 0.25, 0.5)
        line._objs = [char1, char2, char3, char4]
        # keeping to powers of 2
        # to avoid machine specific FP malarchy

        mock_miner["device"].return_value.cur_item = line

        ret = content_lines("test/path")
        assert ret[0][0] == (0.25, 0.25, 0.75, 0.75, "Test")


def test_page_cont_extraction_char(mock_miner):
    with mock.patch("builtins.open", mock.mock_open(read_data=b"this is a test")):
        mock_miner["page"].create_pages.return_value = [dummy_page()]
        line = LTTextLine(0.3)
        line.bbox = (0.25, 0.25, 0.75, 0.75)
        char1 = char("T", 0.25, 0.5, 0.25, 0.5)
        char2 = char("e", 0.25, 0.5, 0.25, 0.5)
        char3 = char("s", 0.25, 0.5, 0.25, 0.5)
        char4 = char("t", 0.25, 0.5, 0.25, 0.5)
        line._objs = [char1, char2, char3, char4]
        mock_miner["device"].return_value.cur_item = line

        ret = content_chars("test/path")

        assert ret[0][0] == (0.5, 0.25, 0.75, 0.5, "T")
        assert ret[0][1] == (0.5, 0.25, 0.75, 0.5, "e")
        assert ret[0][2] == (0.5, 0.25, 0.75, 0.5, "s")
        assert ret[0][3] == (0.5, 0.25, 0.75, 0.5, "t")
        assert len(ret[0]) == 4


def test_page_cont_discard_oob(mock_miner):
    with mock.patch("builtins.open", mock.mock_open(read_data=b"this is a test")):
        page = dummy_page()
        page.cropbox = (0.0, 0.0, 0.8, 0.8)
        mock_miner["page"].create_pages.return_value = [page]
        line = LTTextLine(0.3)
        line.bbox = (0.25, 0.25, 0.75, 0.75)
        char1 = char("T", 0.25, 0.5, 0.25, 0.5)
        char2 = char("e", 0.25, 0.5, 0.25, 0.5)
        char3 = char("s", 0.25, 0.5, 0.25, 0.5)
        char4 = char("t", 0.9, 1.0, 0.25, 0.5)
        line._objs = [char1, char2, char3, char4]
        mock_miner["device"].return_value.cur_item = line

        ret = content_chars("test/path")

        assert ret[0][0] == (0.5, 0.25, 0.75, 0.5, "T")
        assert ret[0][1] == (0.5, 0.25, 0.75, 0.5, "e")
        assert ret[0][2] == (0.5, 0.25, 0.75, 0.5, "s")
        assert len(ret[0]) == 3
