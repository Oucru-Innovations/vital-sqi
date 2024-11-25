import pandas as pd
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate  # Correctly import PreventUpdate
from vital_sqi.app.app import app


# Layout Definition
layout = html.Div(
    [
        # Input for Adding Columns
        html.Div(
            [
                dcc.Input(
                    id="editing-columns-name",
                    placeholder="Enter a column name...",
                    value="",
                    style={"padding": 10},
                ),
                html.Button("Add Column", id="editing-columns-button", n_clicks=0),
            ],
            style={"height": 50, "margin-bottom": "10px"},
        ),
        # Data Table Display
        html.Div(id="data-table"),
    ]
)


# Callback for Generating and Displaying Data Table
@app.callback(Output("data-table", "children"), Input("dataframe", "data"))
def on_data_set_table(data):
    """
    Callback to display and manipulate the data table.
    Processes the uploaded DataFrame and generates an editable table.
    """
    try:
        if data is None:
            raise PreventUpdate

        # Create the DataFrame
        df = pd.DataFrame(data)

        # Compute summary row with means for numeric columns
        summary_row = df.mean(numeric_only=True).to_dict()
        summary_row.update({col: "Summary" for col in df.columns if col not in summary_row})
        df_summary = pd.DataFrame([summary_row])

        # Concatenate the original DataFrame with the summary row
        df_combined = pd.concat([df, df_summary], ignore_index=True)

        # Define DataTable for Display
        table = dash_table.DataTable(
            id="editing-columns",
            columns=[{"name": col, "id": col, "deletable": True} for col in df_combined.columns],
            data=df_combined.to_dict("records"),
            style_table={"overflowX": "auto"},
            editable=True,
            style_cell={
                "textAlign": "left",
                "minWidth": "110px",
                "width": "110px",
                "maxWidth": "110px",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
            },
            tooltip_data=[
                {
                    column: {"value": str(value), "type": "markdown"}
                    for column, value in row.items()
                }
                for row in df_combined.to_dict("records")
            ],
            tooltip_duration=None,
            filter_action="native",
            sort_action="native",
            sort_mode="single",
            page_action="native",
            page_size=30,
        )
        return table

    except Exception as e:
        print(f"Error in on_data_set_table: {e}")
        return html.Div("An error occurred while processing the data.")
