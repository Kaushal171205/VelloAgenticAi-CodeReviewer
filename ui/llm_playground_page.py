"""
llm_playground_page.py — Interactive LLM Provider Abstraction Playground.

Provides UI for testing Gemini and Ollama providers, checking connectivity,
verifying response generation, inspecting token counts & latency, and testing
schema-constrained structured outputs without exposing sensitive API credentials.
"""

from __future__ import annotations

import json
from typing import Optional

import streamlit as st
from pydantic import BaseModel, Field

from config import LLMProvider, settings
from llm import BaseLLMProvider, get_llm_provider


# Sample Pydantic model for structured output testing in the UI
class CodeReviewSample(BaseModel):
    has_issues: bool = Field(description="True if security or logic issues detected")
    severity: str = Field(description="Highest severity level (HIGH, MEDIUM, LOW, NONE)")
    summary: str = Field(description="Brief summary of code review findings")
    suggested_fix: Optional[str] = Field(default=None, description="Suggested fix if applicable")


def render_llm_playground_page() -> None:
    """Render Phase 4 — LLM Provider Abstraction Playground page."""
    st.markdown(
        """
        <div class="main-header">
            <h1>🤖 LLM Provider Abstraction</h1>
            <p>Test and verify interchangeable LLM backends (Gemini & Ollama) with unified prompt execution, token usage tracking, and structured output parsing.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background-color: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 8px; padding: 12px 16px; margin-bottom: 24px; font-size: 0.9rem;">
            🔒 <strong>Secret Sanitization Guarantee:</strong> All API keys are masked using safe truncation (e.g. <code>AQ.A...vbw</code>) before rendering or logging.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_config, col_status = st.columns([1, 1], gap="large")

    with col_config:
        st.subheader("⚙️ Provider Configuration")
        selected_provider_str = st.selectbox(
            "Select Active LLM Provider",
            options=[p.value for p in LLMProvider],
            index=0 if settings.llm_provider == LLMProvider.GEMINI else 1,
            help="Choose between Google Gemini API or local Ollama server.",
        )

        try:
            provider: BaseLLMProvider = get_llm_provider(selected_provider_str)
        except Exception as e:
            st.error(f"Failed to instantiate provider: {e}")
            return

        st.info(
            f"**Active Model:** `{provider.model_name}`\n\n"
            f"**Backend:** `{provider.provider_name.upper()}`"
        )

    with col_status:
        st.subheader("📡 Backend Health Status")
        if st.button("🔄 Run Health Check", use_container_width=True, type="secondary"):
            with st.spinner(f"Pinging {provider.provider_name.upper()}..."):
                health = provider.health_check()
                st.session_state[f"health_{provider.provider_name}"] = health

        health_cached = st.session_state.get(f"health_{provider.provider_name}")
        if health_cached:
            if health_cached.is_healthy:
                st.success(
                    f"✅ **Healthy**\n\n"
                    f"**Message:** {health_cached.status_message}\n\n"
                    f"**Latency:** {health_cached.latency_ms} ms"
                )
            else:
                st.error(
                    f"❌ **Unhealthy**\n\n"
                    f"**Message:** {health_cached.status_message}"
                )
        else:
            st.caption("Click 'Run Health Check' to verify connection and model readiness.")

    st.markdown("---")

    # ── Interactive Prompt Testing Section ──────────────────────────────────
    st.subheader("🧪 Interactive Prompt Execution")

    tab_text, tab_structured = st.tabs(["💬 Text Generation", "📐 Structured Output (JSON)"])

    # ── Tab 1: Text Generation ──────────────────────────────────────────────
    with tab_text:
        col_in, col_opts = st.columns([2, 1], gap="medium")
        with col_in:
            prompt_input = st.text_area(
                "Prompt Input",
                value="Explain why secret sanitization is crucial before passing enterprise code to an LLM in 2 paragraphs.",
                height=130,
            )
            system_prompt_input = st.text_input(
                "System Prompt (Optional)",
                value="You are an expert enterprise application security engineer.",
            )

        with col_opts:
            temp_input = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
            max_tokens_input = st.number_input("Max Output Tokens", min_value=10, max_value=4096, value=512)

        if st.button("🚀 Generate Text Response", type="primary", use_container_width=True):
            if not prompt_input.strip():
                st.warning("Please enter a prompt.")
            else:
                with st.spinner("Generating completion..."):
                    res = provider.generate(
                        prompt=prompt_input,
                        system_prompt=system_prompt_input if system_prompt_input.strip() else None,
                        temperature=temp_input,
                        max_tokens=int(max_tokens_input),
                    )

                if res.status == "success":
                    st.success(f"Generated response in **{res.latency_ms:.2f} ms**")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Prompt Tokens", res.usage.prompt_tokens)
                    m2.metric("Completion Tokens", res.usage.completion_tokens)
                    m3.metric("Total Tokens", res.usage.total_tokens)

                    st.markdown("### Output Text")
                    st.write(res.content)
                else:
                    st.error(f"Generation failed: {res.error_message}")

    # ── Tab 2: Structured Output ────────────────────────────────────────────
    with tab_structured:
        st.markdown(
            "Test constrained structured JSON generation conforming to a Pydantic schema (`CodeReviewSample`)."
        )

        code_snippet_input = st.text_area(
            "Code Snippet to Evaluate",
            value="""import os\nuser_input = input()\nos.system(f"ping {user_input}")""",
            height=100,
        )

        if st.button("📐 Generate Structured Output", type="primary", use_container_width=True):
            with st.spinner("Generating schema-validated JSON..."):
                struct_res = provider.generate_structured(
                    prompt=f"Perform a preliminary security evaluation of this code:\n```\n{code_snippet_input}\n```",
                    schema=CodeReviewSample,
                    system_prompt="Return a structured JSON review adhering to the specified schema.",
                )

            if struct_res.is_valid:
                st.success("✅ Output successfully parsed & validated against Pydantic schema!")

                col_json, col_model = st.columns(2)
                with col_json:
                    st.markdown("#### Validated Pydantic Object")
                    st.json(struct_res.parsed_data.model_dump())

                with col_model:
                    st.markdown("#### Raw LLM Response")
                    st.code(struct_res.raw_text, language="json")
            else:
                st.error(f"❌ Structured output validation failed: {struct_res.error_message}")
                st.markdown("#### Raw Response Text")
                st.code(struct_res.raw_text)


# Standard page render alias
render = render_llm_playground_page
