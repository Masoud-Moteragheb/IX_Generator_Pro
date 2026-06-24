# IX Generator Pro - Final Clean

## Changes

1. No curves are selected initially.
2. Curve selection becomes available after uploading the Excel file.
3. Fixed GPR/noise message removed from UI.
4. Generation mode is limited to Research and Smooth.
5. Internal BV density is always Auto and hidden from user.
6. Summary metrics section removed.
7. Values below 1 are exported and displayed as 0.
8. Excel output has only five columns per sheet: BV, Co, Cu, Co_C_C0, Cu_C_C0.
9. BV is integer; all other numeric values are rounded to two decimals.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
