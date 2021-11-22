import argparse
import logging
import os
import shutil
import tempfile
import uuid

import pandas as pd
import smart_open
import xlsxwriter

from onai.ml.peers.candidate_suggestion.albert import AlbertRankNetCandidateSuggestion
from onai.ml.peers.candidate_suggestion.albert import Config as AlbertCfg
from onai.ml.peers.candidate_suggestion.bert import BertRankNetCandidateSuggestion
from onai.ml.peers.candidate_suggestion.bert import Config as BertCfg
from onai.ml.peers.candidate_suggestion.es import Config as ESCSCfg
from onai.ml.peers.candidate_suggestion.es import ESCandidateSuggestion
from onai.ml.peers.candidate_suggestion.random import Config as RandomCSCfg
from onai.ml.peers.candidate_suggestion.random import RandomCandidateSuggestion
from onai.ml.peers.candidate_suggestion.ranknet import Config as RankNetCfg
from onai.ml.peers.candidate_suggestion.ranknet import RankNetCandidateSuggestion
from onai.ml.peers.experiment.evaluate_consensus import (
    COMPANY_COL_NAME,
    POS_NEG_COL_NAME,
    REASON_1_COL_NAME,
    REASON_2_COL_NAME,
    REASON_3_COL_NAME,
    REL_POS_COL_NAME,
)
from onai.ml.peers.feature_extractor import last_reported_financial
from onai.ml.peers.types import CompanyDetail, Financial
from onai.ml.tools.argparse import add_bool_argument, extract_subgroup_args
from onai.ml.tools.logging import _clean_hdlrs, setup_logger

logger = logging.getLogger(__name__)


def format_financial(f: Financial):
    if f is None:
        return "N/A"
    ret = ""
    if f.currency:
        ret = f.currency + " "
    if f.val:
        ret += f"{f.val:.4f}"
    else:
        return "N/A"
    if f.magnitude:
        ret += f.magnitude
    return ret


def generate_xlsx_for_base_company(base_company, company_idx, cs, uid, args):
    try:
        base_company_detail, peers = cs.suggest_candidates_by_name(
            base_company, args.start_year, args.end_year, args.p
        )

        with tempfile.TemporaryDirectory() as d:

            wb = xlsxwriter.Workbook(os.path.join(d, "out.xlsx"))
            ws_1 = wb.add_worksheet(name="Output Peers Annotation")
            fill_annotation_sheet(base_company_detail, peers, wb, ws_1)

            ws_2 = wb.add_worksheet(name="Auxiliary Information")
            fill_aux_info_sheet(base_company_detail, company_idx, peers, wb, ws_2)

            ws_3 = wb.add_worksheet(name="Peer Financials")
            fill_peer_financials_sheet(
                base_company_detail, company_idx, peers, wb, ws_3
            )

            ws_4 = wb.add_worksheet(name="peer_entity_ids")

            fill_peer_entity_ids_sheet(peers, wb, ws_4)

            wb.close()
            for i in range(args.c):
                with smart_open.open(wb.filename, "rb") as fin, smart_open.open(
                    os.path.join(
                        args.o, f"Peer_Annotation_B{company_idx}_{uid}_{i}.xlsx"
                    ),
                    "wb",
                ) as fout:
                    shutil.copyfileobj(fin, fout)
    except Exception as e:
        logger.exception(f"Skipping {base_company} due to: {e}")


def fill_annotation_sheet(base_company_detail, peers, wb, ws):
    ws.write(0, 4, "Last reported financials in the last 3 years")
    last_year_financials = sorted(base_company_detail.financials.keys())
    header = (
        [COMPANY_COL_NAME, "Description", "Region", "Country"]
        + last_year_financials
        + [
            POS_NEG_COL_NAME,
            REASON_1_COL_NAME,
            REASON_2_COL_NAME,
            REASON_3_COL_NAME,
            REL_POS_COL_NAME,
        ]
    )
    ws.write_row(1, 1, header)
    header_to_column_idx = {k: v for v, k in enumerate(header, 1)}
    peers = sorted(peers, key=lambda x: x.rank)
    last_year_financials_s = set(last_year_financials)
    years_of_financials = [
        x.year for x in next(iter(base_company_detail.financials.values()))
    ]
    for p in peers:
        # check if financial items are the same across peers
        assert set(p.detail.financials.keys()) == last_year_financials_s, (
            last_year_financials_s,
            p.detail.financials,
        )
        for v in p.detail.financials.values():
            # check financials are sorted by year
            assert all(v[i].year <= v[i + 1].year for i in range(len(v) - 1))
            # check if different financials have the same amount of values
            assert [x.year for x in v] == years_of_financials

    financials_cell_fmt = wb.add_format(
        {
            "font_name": "Arial",
            "font_size": 10,
            "valign": "vcenter",
            "align": "center",
            "bg_color": "#819FF7",
        }
    )
    annotation_cell_fmt = wb.add_format(
        {
            "font_name": "Arial",
            "font_size": 10,
            "valign": "vcenter",
            "align": "left",
            "bg_color": "#CEF6D8",
        }
    )

    neg_annotation_cell_fmt = wb.add_format(
        {
            "font_name": "Arial",
            "font_size": 10,
            "font_color": "#FF1100",
            "valign": "vcenter",
            "align": "left",
            "bg_color": "#CEF6D8",
        }
    )

    def render_detail(row, peer_detail):
        financials = [
            format_financial(last_reported_financial(peer_detail.financials, k))
            for k in last_year_financials
        ]
        ws.write_row(
            row,
            header_to_column_idx[COMPANY_COL_NAME],
            [
                peer_detail.name,
                peer_detail.description,
                peer_detail.region,
                peer_detail.country,
            ],
        )

        ws.write_row(
            row,
            header_to_column_idx[last_year_financials[0]],
            financials,
            financials_cell_fmt,
        )
        # remaining columns formatting
        ws.write_row(
            row,
            header_to_column_idx[POS_NEG_COL_NAME],
            [
                ""
                for _ in range(
                    header_to_column_idx[REL_POS_COL_NAME]
                    - header_to_column_idx[POS_NEG_COL_NAME]
                    + 1
                )
            ],
            cell_format=annotation_cell_fmt,
        )

    # the 2nd row is base borrower
    render_detail(2, base_company_detail)
    # from the third row onwards, it is peers suggestion
    for row, p in enumerate(peers, 3):
        render_detail(row, p.detail)
    # company column format
    cell_fmt = wb.add_format(
        {"text_wrap": True, "font_name": "Arial", "font_size": 10, "valign": "vcenter"}
    )
    ws.set_column(
        header_to_column_idx[COMPANY_COL_NAME],
        header_to_column_idx[COMPANY_COL_NAME],
        width=20,
        cell_format=cell_fmt,
    )
    # description column formatting
    ws.set_column(
        header_to_column_idx["Description"],
        header_to_column_idx["Description"],
        width=100,
        cell_format=cell_fmt,
    )
    # header row formatting
    cell_fmt = wb.add_format(
        {
            "font_name": "Arial",
            "font_size": 10,
            "font_color": "#FFFFFF",
            "valign": "vcenter",
            "align": "center",
            "bg_color": "#0B0B3B",
            "text_wrap": True,
        }
    )
    ws.set_row(1, height=25, cell_format=cell_fmt)
    # financial column width formatting
    ws.set_column(
        header_to_column_idx[last_year_financials[0]],
        header_to_column_idx[last_year_financials[-1]],
        width=15,
    )
    # pos/neg col
    ws.set_column(
        header_to_column_idx[POS_NEG_COL_NAME],
        header_to_column_idx[POS_NEG_COL_NAME],
        width=15,
    )
    # reasons col
    ws.set_column(
        header_to_column_idx[REASON_1_COL_NAME],
        header_to_column_idx[REASON_3_COL_NAME],
        width=30,
    )
    # relative pos col
    ws.set_column(
        header_to_column_idx[REL_POS_COL_NAME],
        header_to_column_idx[REL_POS_COL_NAME],
        width=15,
    )
    # data validation on pos/neg col
    ws.data_validation(
        2,
        header_to_column_idx[POS_NEG_COL_NAME],
        2 + len(peers),
        header_to_column_idx[POS_NEG_COL_NAME],
        {"validate": "list", "source": ["Positive", "Negative"]},
    )
    # conditional formatting on pos/neg col

    ws.conditional_format(
        2,
        header_to_column_idx[POS_NEG_COL_NAME],
        2 + len(peers),
        header_to_column_idx[POS_NEG_COL_NAME],
        {
            "type": "cell",
            "criteria": "equal to",
            "value": '"Negative"',
            "format": neg_annotation_cell_fmt,
        },
    )
    # data validation on reasons col
    ws.data_validation(
        2,
        header_to_column_idx[REASON_1_COL_NAME],
        2 + len(peers),
        header_to_column_idx[REASON_3_COL_NAME],
        {
            "validate": "list",
            "source": [
                "Sector mismatch",
                "Same sector but business mismatch",
                "Geography mismatch",
                "Revenue range mismatch",
                "Short business description",
                "Subsidiary of same parent",
                "Duplicate",
                "Others",
            ],
        },
    )
    # data validation on rel pos col
    ws.data_validation(
        2,
        header_to_column_idx[REL_POS_COL_NAME],
        2 + len(peers),
        header_to_column_idx[REL_POS_COL_NAME],
        {"validate": "list", "source": ["Most relevant", "Relevant", "Least relevant"]},
    )
    # freeze the second row and second column
    ws.freeze_panes(2, 2)


def fill_aux_info_sheet(base_company_detail, company_idx, peers, wb, ws):
    # header row for base borrower section
    cell_fmt = wb.add_format(
        {
            "font_name": "Arial",
            "font_size": 10,
            "valign": "vcenter",
            "align": "left",
            "bg_color": "#CEECF5",
        }
    )
    ws.write_row(
        1,
        1,
        [
            "Base Borrower ID",
            "Base Borrower Name",
            "Base Borrower Description",
            "Base Borrower Region",
            "Base Borrower Revenue",
        ],
        cell_format=cell_fmt,
    )
    last_reported_total_rev = None
    if "TOTAL_REVENUE" in base_company_detail.financials:
        last_reported_total_rev = base_company_detail.financials["TOTAL_REVENUE"][-1]

    ws.write_row(
        2,
        1,
        [
            company_idx,
            base_company_detail.name,
            base_company_detail.description,
            base_company_detail.region,
            format_financial(last_reported_total_rev),
        ],
    )

    # last three years report table
    # table header row
    n_year_look_backwd = 4
    last_reported_yrs = [
        x.year for x in next(iter(base_company_detail.financials.values()))
    ][-n_year_look_backwd:]

    fye_month_str = (
        base_company_detail.fye.strftime("%b") if base_company_detail.fye else ""
    )

    cell_fmt = wb.add_format(
        {
            "bold": True,
            "font_name": "Arial",
            "font_size": 9,
            "valign": "vcenter",
            "align": "left",
            "top": True,
            "bottom": True,
            "right": True,
        }
    )

    ws.write_string(
        4, 2, f"{base_company_detail.name} Financials", cell_format=cell_fmt
    )
    cell_fmt = wb.add_format(
        {
            "bold": True,
            "font_name": "Arial",
            "font_size": 9,
            "valign": "vcenter",
            "align": "right",
            "top": True,
            "bottom": True,
        }
    )
    ws.write_row(
        4,
        3,
        [f"{fye_month_str}'{yr}" for yr in last_reported_yrs],
        cell_format=cell_fmt,
    )

    last_year_financials = sorted(base_company_detail.financials.keys())

    cell_fmt = wb.add_format(
        {
            "bold": True,
            "font_name": "Arial",
            "font_size": 9,
            "valign": "vcenter",
            "align": "left",
            "right": True,
        }
    )
    ws.write_column(5, 2, last_year_financials, cell_format=cell_fmt)
    base_borrower_financial_cell_fmt = wb.add_format(
        {"font_name": "Arial", "font_size": 10, "align": "right"}
    )
    for row_idx, k in enumerate(last_year_financials, 5):
        financials = base_company_detail.financials[k][-n_year_look_backwd:]
        ws.write_row(
            row_idx,
            3,
            [format_financial(f) for f in financials],
            cell_format=base_borrower_financial_cell_fmt,
        )

    ws.write_row(
        row_idx + 1,
        2,
        [
            ""
            for x in range(
                len(
                    base_company_detail.financials[last_year_financials[0]][
                        -n_year_look_backwd:
                    ]
                )
                + 1
            )
        ],
        wb.add_format({"top": True}),
    )

    base_borrower_end_row_idx = row_idx
    peer_section_start_idx = base_borrower_end_row_idx + 2

    ws.write(peer_section_start_idx, 1, "Peers")
    ws.write_row(
        peer_section_start_idx + 1,
        1,
        [
            "Base Borrower ID",
            "Peer Company Name",
            "Peer Description",
            "Peer Region",
            "Peer Financial Year End (MM-DD)",
        ],
        wb.add_format(
            {
                "font_name": "Arial",
                "font_size": 10,
                "valign": "vcenter",
                "align": "left",
                "bg_color": "#CEECF5",
            }
        ),
    )
    peer_details_cell_fmt = wb.add_format(
        {"font_name": "Arial", "font_size": 10, "valign": "bottom", "align": "left"}
    )
    for row_idx, p in enumerate(peers, peer_section_start_idx + 2):
        ws.write_row(
            row_idx,
            1,
            [
                company_idx,
                p.detail.name,
                p.detail.description,
                p.detail.region,
                p.detail.fye.strftime("%m-%d") if p.detail.fye else None,
            ],
            cell_format=peer_details_cell_fmt,
        )

    # formatting column widths.
    ws.set_column(1, 1, width=15)
    ws.set_column(2, 2, width=25)
    # description requires text wrap otherwise it will be unbearable to read
    cell_fmt = wb.add_format(
        {
            "font_name": "Arial",
            "font_size": 10,
            "valign": "vcenter",
            "align": "left",
            "text_wrap": True,
        }
    )
    ws.set_column(3, 3, width=130, cell_format=cell_fmt)
    ws.set_column(4, 4, width=20)
    ws.set_column(5, 5, width=20)


def fill_peer_financials_sheet(base_company_detail, company_idx, peers, wb, ws):
    yrs = [x.year for x in next(iter(base_company_detail.financials.values()))]
    last_year_financials = sorted(base_company_detail.financials.keys())
    row_idx = 0
    header_fmt = wb.add_format({"top": True})
    for k in last_year_financials:
        ws.write_string(row_idx, 0, k)
        ws.write_row(
            row_idx + 1,
            0,
            ["Base Borrower ID", "Peer Name"] + yrs,
            cell_format=header_fmt,
        )
        row_idx += 2
        for p in peers:
            ws.write_row(
                row_idx,
                0,
                [company_idx, p.detail.name]
                + [format_financial(f) for f in p.detail.financials[k]],
            )
            row_idx += 1

        row_idx += 1

    # set base borrower ID column
    ws.set_column(0, 0, width=20)
    # set peer name column
    ws.set_column(1, 1, width=30, cell_format=wb.add_format({"right": True}))


# TODO: entity id of base borrower should also be populated here (when available)
def fill_peer_entity_ids_sheet(peers, wb, ws):
    ws.write_string(
        0,
        0,
        "DO NOT MODIFY THIS SHEET. YOUR ANNOTATION WILL BE INVALIDATED OTHERWISE.",
        cell_format=wb.add_format({"font_color": "red"}),
    )
    for row_idx, p in enumerate(peers, 1):
        detail: CompanyDetail = p.detail
        ws.write_row(row_idx, 0, [detail.name, detail.entity_id])


def main():
    _clean_hdlrs()
    setup_logger()
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", help="the base borrower file. aka internal_ds", required=True
    )
    parser.add_argument(
        "-w",
        nargs="*",
        help="Selectively only generate annotations for the "
        "companies that are specified with this option",
        type=int,
    )
    parser.add_argument("-o", help="Output folder", required=True)
    parser.add_argument("-c", help="Consensus level", type=int, default=1)
    parser.add_argument(
        "-p", help="How many peers to be generated", type=int, default=20
    )
    parser.add_argument(
        "--start_year",
        help="Which year to consider financials from",
        type=int,
        default=2008,
    )
    parser.add_argument(
        "--end_year",
        help="Which year to consider financials to",
        type=int,
        default=2019,
    )
    add_bool_argument(parser, "anonymise_base_company", default=True)
    add_bool_argument(parser, "profile", default=False)
    subparser = parser.add_subparsers(
        title="which baseline to use", help="eg. random, es", dest="baseline_type"
    )

    random_pg = subparser.add_parser("random").add_argument_group()
    RandomCSCfg.populate_argparser(random_pg)

    es_pg = subparser.add_parser("es").add_argument_group()
    ESCSCfg.populate_argparser(es_pg)

    ranknet_pg = subparser.add_parser("ranknet").add_argument_group()
    RankNetCfg.populate_argparser(ranknet_pg)

    albert_ranknet_pg = subparser.add_parser("albert_ranknet").add_argument_group()
    AlbertCfg.populate_argparser(albert_ranknet_pg)

    bert_ranknet_pg = subparser.add_parser("bert_ranknet").add_argument_group()
    BertCfg.populate_argparser(bert_ranknet_pg)

    args = parser.parse_args()

    if args.baseline_type == "random":
        cfg = RandomCSCfg.from_dict(extract_subgroup_args(args, random_pg))
        cs = RandomCandidateSuggestion(cfg)
    elif args.baseline_type == "es":
        pg_args = extract_subgroup_args(args, es_pg)
        pg_args["internal_ds"] = args.i
        cfg = ESCSCfg.from_dict(pg_args)
        cs = ESCandidateSuggestion.from_cfg(cfg)
    elif args.baseline_type == "ranknet":
        pg_args = extract_subgroup_args(args, ranknet_pg)
        pg_args["internal_ds"] = args.i
        cfg = RankNetCfg.from_dict(pg_args)
        cs = RankNetCandidateSuggestion.from_cfg(cfg)
    elif args.baseline_type == "albert_ranknet":
        pg_args = extract_subgroup_args(args, albert_ranknet_pg)
        pg_args["internal_ds"] = args.i
        cfg = AlbertCfg.from_dict(pg_args)
        cs = AlbertRankNetCandidateSuggestion.from_cfg(cfg)
    elif args.baseline_type == "bert_ranknet":
        pg_args = extract_subgroup_args(args, bert_ranknet_pg)
        pg_args["internal_ds"] = args.i
        cfg = BertCfg.from_dict(pg_args)
        cs = BertRankNetCandidateSuggestion.from_cfg(cfg)
    else:
        assert False, "Unknown type of baseline. Only supporting random now"

    uid = uuid.uuid1()
    base_companies = pd.read_excel(args.i, ["Info", "Financials"])
    white_list = None
    if args.w:
        white_list = set(args.w)

    import time

    start_t = None
    count = 0
    pr = None
    if args.profile:
        import cProfile

        pr = cProfile.Profile()
    for _, base_company_row in base_companies["Info"].iterrows():
        if not white_list or base_company_row["Case"] in white_list:
            generate_xlsx_for_base_company(
                base_company_row["Base Borrower Name"],
                base_company_row["Case"],
                cs,
                uid,
                args,
            )
            count += 1
            # we don't calculate the first batch as it skews the result significantly.
            if start_t is None:
                start_t = time.time()
                if args.profile:
                    pr.enable()
    if args.profile:
        pr.disable()
        pr.dump_stats("/dev/shm/bert.benchmark")
    if start_t is not None and count > 1:
        logger.info(f"Total spent time {time.time() - start_t:.4f}s")
        logger.info(f"{(time.time() - start_t) / (count - 1):.4f} s/company")


if __name__ == "__main__":
    main()
