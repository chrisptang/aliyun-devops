import pandas as pd

css = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
<style type=\"text/css\">
#abTesting table,#abTesting .table {
    color: #333;
    font-family: unset;
    font-size: 12px;
    line-height: 1.5;
    width: 90vw;
    border-collapse:
    collapse; 
    border-spacing: 0;
    font-family: "SF Pro SC", "SF Pro Text", "SF Pro Icons", "PingFang SC", "Helvetica Neue", "Helvetica", "Arial", sans-serif;
}

body{
    padding-left: 2rem;
    padding-top: 1vh;
}

tr{
    border-bottom: 1px solid #C1C3D1;
}

tr:nth-child(even) {
    background-color: #F8F8F8;
}

#abTesting td, #abTesting th {
    /* border: 1px solid transparent; No more visible border */
    height: 30px;
    padding: 0.5rem;
}

#abTesting table tbody td,#abTesting .table tbody td{
    padding: 0.1rem .75rem;
    vertical-align: middle;
}

th {
    background-color: #DFDFDF; /* Darken header a bit */
    font-weight: bolder;
    font-size: larger;
    color: #000;
    text-align: center;
}
</style>
"""


def display_p_value_below_005(
    row: pd.Series, p_value_col_name: str = "p_value", theshold: float = 0.05
):
    p_value = row[p_value_col_name]
    color = "black"
    if p_value is not None and p_value <= theshold:
        color = "red"
    return f"""<span style='font-weight:bolder;color:{color};'>{p_value}</span>"""


def display_diff_to_v1(row: pd.Series, metric: str = "diff_to_V1_%"):
    diff = row[metric]
    color = "green"
    if diff is not None and diff > 0:
        color = "red"
    return f"""<span style='font-weight:bolder;color:{color};'>{diff:.4f} %</span>"""
