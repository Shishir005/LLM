import streamlit as st
import pandas as pd
import io
import re
import ast
import os
from openai import OpenAI
from difflib import get_close_matches

# Set API Key
os.environ["OPENAI_API_KEY"] = "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"  # Replace with your key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="📊 LLM Rule Engine", layout="wide")
st.title("🤖 LLM + Pandas Rule Engine (with Normalization & Prompt Refinement)")

# --- Upload Excel ---
if "updated_excel" in st.session_state:
    uploaded_file = io.BytesIO(st.session_state["updated_excel"])
else:
    uploaded_file = st.file_uploader("📄 Upload Excel file with 'data' and 'rules' sheets", type=["xlsx"])

# --- Helpers ---
def normalize_dataframe(df):
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
    return df

def correct_column_typos(text, actual_columns):
    words = text.replace("'", "").replace('"', "").replace("_", " ").split()
    fixed = text
    warnings = []
    for word in words:
        matches = get_close_matches(word.lower(), [col.lower() for col in actual_columns], n=1, cutoff=0.8)
        if matches:
            best_match = next((col for col in actual_columns if col.lower() == matches[0]), word)
            fixed = fixed.replace(word, best_match)
            if best_match != word:
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
    if "```" in content:
        match = re.search(r"```(?:python)?\s*(.*?)```", content, re.DOTALL)
        return match.group(1).strip() if match else content.strip()
    return content.strip()

def sanitize_code(code):
    code = re.sub(r"pd\.read_excel\(.*?\)", "df", code)
    code = re.sub(r"df\.to_excel\(.*?\)", "", code)
    return code

# ✅ LLM-Based Prompt Refiner
def refine_prompt(original_prompt: str, df: pd.DataFrame) -> str:
    schema = "\n".join([f"- {col}: {str(df[col].dtype)}" for col in df.columns])
    prompt = f"""
You are a prompt improvement assistant.
For the specified columns in the DataFrame, convert all values to strings, strip leading/trailing whitespace, and convert them to lowercase.
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
    df.columns = df.columns.str.strip().str.lower()
    columns = list(df.columns)

    # Step 1: Fix typos in column names
    fixed_text, warnings = correct_column_typos(rule_text, columns)
    for w in warnings:
        st.warning(w)

    # Step 2: Improve rule via LLM
    improved_text = refine_prompt(fixed_text, df)
    st.info(f"🤖 Refined Prompt: {improved_text}")

    # Step 3: Create LLM prompt
    prompt = f"""
    prompt = (
        f"You are a Pandas expert. The DataFrame `df` has the following lowercase column names:\n"
        f"{columns}\n\n"
        f"Write Python code that modifies `df` according to this instruction:\n"
        f"\"\"\"{improved_text}\"\"\"\n\n"
        f"Use only the column names exactly as listed. Do not use file I/O functions."
    )
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        code = sanitize_code(extract_python_code(response.choices[0].message.content))
        st.info("🔢 LLM Code:")
        st.code(code, language="python")

        used_cols = extract_column_names_from_code(code)
        if not used_cols.issubset(df.columns):
            st.error(f"⛔ Unknown columns used: {used_cols - set(df.columns)}")
            return df

        local_env = {"df": df.copy()}
        exec(code, {}, local_env)
        return local_env["df"]

    except Exception as e:
        st.error(f"❌ Rule failed: {e}")
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
            with st.expander("🧪 Debug Columns", expanded=False):
                col_to_inspect = st.selectbox("🔍 Select column to inspect:", options=data_df.columns)
                unique_vals = sorted(data_df[col_to_inspect].dropna().unique())
                st.write(f"🔢 Unique values in `{col_to_inspect}`:")
                st.code(unique_vals)

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

    selected_apply_index = st.radio(
        "✅ Select ONE rule to apply:",
        options=[None] + list(range(len(rules_df))),
        format_func=format_label,
        key="selected_apply_rule"
    )

    with st.expander("📝 Edit Rules", expanded=True):
        for i, row in rules_df.iterrows():
            rule = st.text_area(f"🧠 Rule {i+1}", value=row["Rule_Description"], key=f"rule_{i}", height=68)
            edited_rules.append(rule)
            rule_actions.append("Apply" if i == selected_apply_index else "Skip")

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
            result_df = apply_llm_rule(result_df, rule)

            #st.write('result_df',result_df)

    st.subheader("✅ Result")
    st.dataframe(result_df)
    st.write(f"Total Records: {len(result_df)}")

    output_final = io.BytesIO()
    result_df.to_excel(output_final, index=False)
    st.download_button("📥 Download Final Output", data=output_final.getvalue(), file_name="final_result.xlsx")

else:
    st.info("📄 Please upload an Excel file with 'data' and 'rules' sheets.")
