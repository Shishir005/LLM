import streamlit as st
import pandas as pd
import io
import re
import ast
import os
from openai import OpenAI
from difflib import get_close_matches
from langgraph.graph import StateGraph, END
from typing import TypedDict
import pandas as pd
import io
import numpy as np
import re
import datetime as dt



os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "Rules Agent Workflow"


def convert_to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
def standardize_col_names(df):
    """Standardize DataFrame column names to snake_case and remove special chars."""
    df.columns = df.columns.str.replace('[^a-zA-Z0-9 ]', '', regex=True)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    return df
def convert_to_snake_case(name):
    """Convert a string to snake_case"""
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
# Set API Key
os.environ["OPENAI_API_KEY"] = "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"  # Replace with your key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#st.set_page_config(page_title="📊 LLM Rule Engine", layout="wide")
#st.title("🤖 LLM + Pandas Rule Engine (with Normalization & Prompt Refinement)")


# ----- Define State -----
class RuleEngineState(TypedDict):
    file: bytes
    data_df: pd.DataFrame
    rules_df: pd.DataFrame
    output_df: pd.DataFrame

# ----- Nodes -----
def read_excel_node(state: RuleEngineState):
    file_bytes = state["file"]
    excel_file = io.BytesIO(file_bytes)
    state["data_df"] = pd.read_excel(excel_file, sheet_name="data")
    state["rules_df"] = pd.read_excel(excel_file, sheet_name="rules")
    return state

def apply_rules_node(state: RuleEngineState):
    df = state["data_df"]
    rules = state["rules_df"]["Rule_Description"].tolist()

    # Example: apply first rule to filter data
    for rule in rules:
        # In practice, call your apply_llm_rule(df, rule) function here
        df = df[df[df.columns[0]] != ""]  # Dummy example
    state["output_df"] = df
    return state

def output_node(state: RuleEngineState):
    print("✅ Final Output Rows:", len(state["output_df"]))
    return state

# ----- Build Graph -----

















# --- Upload Excel ---

if "updated_excel" in st.session_state:
    uploaded_file = io.BytesIO(st.session_state["updated_excel"])

else:
    uploaded_file = st.file_uploader("📄 Upload Excel file with 'data' and 'rules' sheets", type=["xlsx"])


# --- Helpers ---
def normalize_dataframe(df):
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].str.replace(r"\s+", " ", regex=True)
    df.columns = df.columns.str.strip().str.lower()
    return df

def correct_column_typos(text, actual_columns):
    words = re.findall(r'\b\w+\b', text)
    fixed = text
    warnings = []
    for word in words:
        matches = get_close_matches(word.lower(), [col.lower() for col in actual_columns], n=1, cutoff=0.8)
        if matches and matches[0] != word.lower():
            best_match = next((col for col in actual_columns if col.lower() == matches[0]), word)
            fixed = re.sub(rf'\b{word}\b', best_match, fixed)
            warnings.append(f"⚠️ Did you mean `{best_match}` instead of `{word}`?")
    return fixed, warnings

def extract_column_names_from_code(code):
    try:
        tree = ast.parse(code)
        return {node.slice.value for node in ast.walk(tree)
                if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)}
    except:
        return set()

def extract_python_code(content):
    match = re.search(r"```(?:python)?\s*(.*?)```", content, re.DOTALL)
    return match.group(1).strip() if match else content.strip()

def sanitize_code(code):

    #st.write('Shishir',code)
    code = re.sub(r"pd\.read_excel\(.*?\)", "df", code)
    code = re.sub(r"df\.to_excel\(.*?\)", "", code)
    return code

# ✅ LLM-Based Prompt Refiner





def refine_prompt(original_prompt: str, df: pd.DataFrame) -> str:
    schema = "\n".join([f"- {col}: {str(df[col].dtype)}" for col in df.columns])
    prompt = f"""
You are a prompt improvement assistant.
For the specified columns in the DataFrame, convert all values to strings, strip leading/trailing whitespace.
{schema}

Improve the clarity and precision of the following instruction so it can be directly used to write accurate Pandas filtering or transformation code:

Instruction:
"{original_prompt}"

Only return the improved version of the instruction. Make it suitable for generating correct Python Pandas code.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Prompt refinement failed: {e}")
        return original_prompt

# ✅ Apply LLM Rule with Hybrid Validation
def apply_llm_rule(df, rule_text):
    #df.columns = df.columns.str.strip().str.lower()
    #st.write('rule_text',rule_text)
    columns = list(df.columns)

    fixed_text, warnings = correct_column_typos(rule_text, columns)
    for w in warnings:
        st.warning(w)

    improved_text = refine_prompt(fixed_text, df)
    st.info(f"🤖 Refined Prompt: {improved_text}")

    prompt = (
        f"You are a Pandas expert. The DataFrame `df` has  column names:\n"
        f"{columns}\n\n"
        f"Write Python code that modifies `df` according to this instruction:\n"
        f"\"\"\"{improved_text}\"\"\"\n\n"
        f"Use only the column names exactly as listed. Do not use file I/O functions."
        f"Assign the result to a variable named df_output"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        code = sanitize_code(extract_python_code(response.choices[0].message.content))
        st.code(code, language="python")

        used_cols = extract_column_names_from_code(code)
        if not used_cols.issubset(set(df.columns)):
            st.error(f"⛔ Unknown columns used: {used_cols - set(df.columns)}")
            return df

        #local_env = {"df": df.copy()}
        #exec(code, {}, local_env)
        import numpy as np
        df = pd.read_excel(uploaded_file, sheet_name="data")
        # local_env = {
        #     "df": df.copy(),
        #     "pd": pd,
        #     "np": np
        # }

        local_env = {
            "df": df.copy(),
            "pd": pd,
            "np": np,
            "re": re,
            "standardize_col_names": standardize_col_names,
            "convert_to_snake_case": convert_to_snake_case
            # Add more helpers here
        }

        exec(code, {}, local_env)
        unique_titles = local_env.get("df_output")



        return unique_titles

    except Exception as e:
        st.error(f"❌ Error executing rule: {e}")
        return df

# --- Main App ---
if uploaded_file:
    try:
        data_df = pd.read_excel(uploaded_file, sheet_name="data")
        rules_df = pd.read_excel(uploaded_file, sheet_name="rules")
        #data_df = normalize_dataframe(data_df)

        try:
            data_df = pd.read_excel(uploaded_file, sheet_name="data")
            rules_df = pd.read_excel(uploaded_file, sheet_name="rules")
            #data_df = normalize_dataframe(data_df)

            # --- Debugging Tool ---
            # with st.expander("🧪 Debug Columns", expanded=False):
            #     col_to_inspect = st.selectbox("🔍 Select column to inspect:", options=data_df.columns)
            #     unique_vals = sorted(data_df[col_to_inspect].dropna().unique())
            #     st.write(f"🔢 Unique values in `{col_to_inspect}`:")
            #     st.code(unique_vals)

        except Exception as e:
            st.error(f"❌ Failed to read Excel: {e}")
            st.stop()




    except Exception as e:
        st.error(f"❌ Failed to read Excel: {e}")
        st.stop()

    # Rule Editor
    st.markdown("### 🛠️ Edit or Add Rules")
    edited_rules = []
    rule_actions = []

    def format_label(i):
        if i is None:
            return "None"
        row = rules_df.iloc[i]
        rid = row.get("Role_ID", f"R{i+1}")
        desc = row.get("Rule_Description", "")
        return f"{i+1}. [ID: {rid}] - {desc}"


    st.markdown("""
    <div style="max-height:200px; overflow-y:auto; padding-right:10px; border:1px solid #ddd; border-radius:5px;">
    """, unsafe_allow_html=True)

    selected_apply_index = st.radio(
        "✅ Select ONE rule to apply:",
        options=[None] + list(range(len(rules_df))),
        format_func=format_label,
        key="selected_apply_rule"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📝 Edit Rules", expanded=True):
        st.markdown("""
        <div style="max-height:300px; overflow-y:auto; padding-right:10px;">
        """, unsafe_allow_html=True)

        for i, row in rules_df.iterrows():
            rule = st.text_area(
                f"🧠 Rule {i + 1}",
                value=row["Rule_Description"],
                key=f"rule_{i}",
                height=68
            )
            edited_rules.append(rule)
            rule_actions.append("Apply" if i == selected_apply_index else "Skip")

        st.markdown("</div>", unsafe_allow_html=True)

    # Add new rule
    st.markdown("### ➕ Add New Rule")
    new_rule = st.text_area("New Rule", key="new_rule_text", height=80)
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("➕ Add Rule"):
            if new_rule.strip():
                rules_df = pd.concat([rules_df, pd.DataFrame({"Rule_Description": [new_rule.strip()]})], ignore_index=True)
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    data_df.to_excel(writer, sheet_name="data", index=False)
                    rules_df.to_excel(writer, sheet_name="rules", index=False)
                st.session_state["updated_excel"] = output.getvalue()
                st.success("✅ Rule added! Click Refresh.")

    with col2:
        if st.button("🔁 Refresh Rules"):
            st.rerun()

    with col3:
        if st.button("💾 Save All Rules"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                data_df.to_excel(writer, sheet_name="data", index=False)
                pd.DataFrame({"Rule_Description": edited_rules}).to_excel(writer, sheet_name="rules", index=False)
            st.success("✅ Saved successfully.")
            st.download_button("📥 Download Excel", data=output.getvalue(), file_name="rules_updated.xlsx")

    # Apply rule
    st.markdown("### 🧠 Apply Selected Rule")
    result_df = data_df.copy()
    for rule, action in zip(edited_rules, rule_actions):
        if rule.strip() and action == "Apply":
            st.markdown(f"**⚙️ Applying Rule:** `{rule}`")

            #st.write('result_df' , result_df)

            #Experiment
            original_df = pd.read_excel(uploaded_file, sheet_name="data")
            #original_df.columns = original_df.columns.str.strip().str.lower()  # Normalization step
            data_df = original_df.copy()
            #result_df = apply_llm_rule(result_df, rule)
            #st.write('result_df',result_df)
            #st.write('data_df', data_df)
            result_df = apply_llm_rule(data_df, rule)

            #st.write('result_df',result_df)

    st.subheader("✅ Result")
    st.dataframe(result_df)
    st.write(f"Total Records: {len(result_df)}")


    output_final = io.BytesIO()
    result_df.to_excel(output_final, index=False)
    st.download_button("📥 Download Final Output", data=output_final.getvalue(), file_name="final_result.xlsx")

#else:
    #st.info("📄 Please upload an Excel file with 'data' and 'rules' sheets.")


st.markdown("""
    <style>
    /* Scrollable text areas in Edit/Add Rules */
    .stTextArea textarea {
        overflow: auto !important;  /* Enables scrollbars */
        white-space: pre !important; /* Keeps formatting and allows horizontal scroll */
        max-height: 150px;           /* Adjust height for vertical scroll */
    }

    /* Optional: Add custom scroll bar styling */
    .stTextArea textarea::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    .stTextArea textarea::-webkit-scrollbar-thumb {
        background-color: #888;
        border-radius: 4px;
    }
    .stTextArea textarea::-webkit-scrollbar-thumb:hover {
        background-color: #555;
    }
    </style>
""", unsafe_allow_html=True)
