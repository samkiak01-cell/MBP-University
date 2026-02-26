"""
MBP University — Main Streamlit Application
"""

import os
import logging
import streamlit as st
import anthropic

from config import (
    APP_TITLE,
    APP_DESCRIPTION,
    WELCOME_MESSAGE,
    SYSTEM_PROMPT,
    CLAUDE_MODEL,
    MAX_TOKENS,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    DOCUMENTS_DIR,
)
from ingest import ingest_all, build_vector_store
from retriever import search, format_context, format_sources_for_display

# ── Logging (visible in Streamlit Cloud → Manage app → Logs) ────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("mbp_university")

# ── Page Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MBP University",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── White logo as base64 data URI (no external file needed) ─────────────
LOGO_DATA_URI = "data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyODkiIGhlaWdodD0iODAiIHZpZXdCb3g9IjAgMCAyODkgODAiIGZpbGw9Im5vbmUiPgogIDxnIGNsaXAtcGF0aD0idXJsKCNjbGlwMF8yNTRfNDQ0NykiPgogICAgPHBhdGggZD0iTTY5LjQ4ODMgMEg4NC42NzE0QzkyLjQxMTggMCA5Ny44NzczIDMuOTk0OTkgOTcuODc3MyAxMS4wODRDOTcuODc3MyAxNS4wNzkgOTUuOTU2MiAxOC41Mjk5IDkyLjcxNTIgMjAuMjg2N1YyMC40MDU2Qzk3LjM5OTkgMjEuNzk5NyA5OS42MTg2IDI2LjE1NzMgOTkuNjE4NiAzMC41MTQ5Qzk5LjYxODYgMzkuMTExMiA5Mi43NzY5IDQyLjk4NzIgODQuOTEzIDQyLjk4NzJINjkuNDg4M1YwWk04NC43MzMyIDE3LjU2MUM4OC4wOTIyIDE3LjU2MSA5MC4wMTMzIDE1LjE0MTMgOTAuMDEzMyAxMi4wNTNDOTAuMDEzMyA4Ljk2NDY2IDg4LjE1NDEgNi43MjYzMiA4NC42MDk3IDYuNzI2MzJINzcuMjI4N1YxNy41NjY3SDg0LjczMzJWMTcuNTYxWk04NS41MTQgMzYuMjcyM0M4OS40NzQxIDM2LjI3MjMgOTEuNjk4NCAzMy42NzE0IDkxLjY5ODQgMzAuMDMzM0M5MS42OTg0IDI2LjQ2MzQgODkuNDE3OSAyMy44NTY2IDg1LjUxNCAyMy44NTY2SDc3LjIyODdWMzYuMjY2Nkg4NS41MTRWMzYuMjcyM1oiIGZpbGw9IndoaXRlIj48L3BhdGg+CiAgICA8cGF0aCBkPSJNMTIxLjQwOSAyMy42MTgxSDEyMi43MjlWMjMuMTkzQzEyMi43MjkgMTkuMTM1NyAxMjAuMTUxIDE3Ljc0MTcgMTE2Ljk2NiAxNy43NDE3QzExMy45NjcgMTcuNzQxNyAxMTAuOTA1IDE5LjAxMSAxMDguNDQ1IDIwLjcxMUwxMDUuNjI1IDE1LjIwM0MxMDguMTQ3IDEzLjE0NjEgMTEzLjA2OCAxMS4zODk0IDExNy42MjggMTEuMzg5NEMxMjUuNjExIDExLjM4OTQgMTMwLjM1MiAxNS44NzE3IDEzMC4zNTIgMjMuNjgwM1Y0Mi45OTc5SDEyMy4yNjhWNDAuMzk3QzEyMy4yNjggMzkuMTI3NyAxMjMuNDQ4IDM4LjE1ODcgMTIzLjQ0OCAzOC4xNTg3SDEyMy4zM0MxMjEuNDcxIDQxLjQyODMgMTE3LjgwOCA0My43MjkgMTEzLjY2OSA0My43MjlDMTA4LjE0NyA0My43MjkgMTAzLjU4NiA0MC4wOTY2IDEwMy41ODYgMzQuNDY0QzEwMy41ODYgMjUuMTkzMyAxMTQuNjI5IDIzLjYxODEgMTIxLjQwOSAyMy42MTgxWk0xMTUuNzY5IDM3Ljg0N0MxMjAuMDMzIDM3Ljg0NyAxMjIuNzkxIDMzLjQ4OTMgMTIyLjc5MSAyOS43MzIzVjI4Ljg4MjRIMTIxLjQ3MUMxMTcuNjI4IDI4Ljg4MjQgMTExLjE0NiAyOS40MjY0IDExMS4xNDYgMzMuNzg5NkMxMTEuMTQ2IDM1Ljg0NjcgMTEyLjcwOCAzNy44NDcgMTE1Ljc2OSAzNy44NDdaIiBmaWxsPSJ3aGl0ZSI+PC9wYXRoPgogICAgPHBhdGggZD0iTTEzOC4zODkgMzQuMDkwOUMxNDAuNjY5IDM2LjIxMDEgMTQ0LjAyOSAzNy40Nzk1IDE0Ny4wOSAzNy40Nzk1QzE0OS43OTIgMzcuNDc5NSAxNTEuMzU0IDM2LjMyOTIgMTUxLjM1NCAzNC4zOTEyQzE1MS4zNTQgMjkuNTQ2MyAxMzYuMTcgMzAuNzU4OSAxMzYuMTcgMjAuNjQzOUMxMzYuMTcgMTQuNTkxOSAxNDEuNDUxIDExLjM3ODkgMTQ3Ljg3NiAxMS4zNzg5QzE1MS41MzkgMTEuMzc4OSAxNTUuMzggMTIuNDEwMyAxNTguMjAxIDE0Ljg5MjJMMTU1LjM4IDIwLjI4MTJDMTUzLjU3OCAxOC43MDU5IDE1MC4zMzcgMTcuNjE3OSAxNDcuNjQxIDE3LjYxNzlDMTQ1IDE3LjYxNzkgMTQzLjM3NyAxOC42NDkyIDE0My4zNzcgMjAuNTg3MkMxNDMuMzc3IDI1LjQ5NDYgMTU4LjU2IDI0LjEwMDUgMTU4LjU2IDM0LjIwOThDMTU4LjU2IDM5LjY2MTIgMTU0IDQzLjcxODUgMTQ2Ljk3OCA0My43MTg1QzE0Mi43MTUgNDMuNzE4NSAxMzguMTUzIDQyLjI2NzkgMTM0Ljk3NCAzOS4xNzk2TDEzOC4zODkgMzQuMDkwOVoiIGZpbGw9IndoaXRlIj48L3BhdGg+CiAgICA8cGF0aCBkPSJNMTk0LjM0MSAwSDIxMC43ODhDMjE4Ljc3IDAgMjI0LjM1NCA1LjYzMjY2IDIyNC4zNTQgMTMuOTg1M0MyMjQuMzU0IDIyLjM0MzcgMjE4Ljc3IDI4LjE1MiAyMTAuNzg4IDI4LjE1MkgyMDIuMDg3VjQyLjk4NzJIMTk0LjM0N1YwSDE5NC4zNDFaTTIwOS4zNDQgMjEuMzc0NkMyMTMuODQ0IDIxLjM3NDYgMjE2LjQ5IDE4LjQ2NzcgMjE2LjQ5IDEzLjk4NTNDMjE2LjQ5IDkuNTY1MzMgMjEzLjg0OSA2LjcyMDY2IDIwOS40NjggNi43MjA2NkgyMDIuMDg3VjIxLjM3NDZIMjA5LjM0NFoiIGZpbGw9IndoaXRlIj48L3BhdGg+CiAgICA8cGF0aCBkPSJNMjQ0LjkxMyAyMy42MTgxSDI0Ni4yMzJWMjMuMTkzQzI0Ni4yMzIgMTkuMTM1NyAyNDMuNjU1IDE3Ljc0MTcgMjQwLjQ2OSAxNy43NDE3QzIzNy40NyAxNy43NDE3IDIzNC40MDkgMTkuMDExIDIzMS45NDggMjAuNzExTDIyOS4xMjkgMTUuMjAzQzIzMS42NTEgMTMuMTQ2MSAyMzYuNTcxIDExLjM4OTQgMjQxLjEzMiAxMS4zODk0QzI0OS4xMTQgMTEuMzg5NCAyNTMuODU1IDE1Ljg3MTcgMjUzLjg1NSAyMy42ODAzVjQyLjk5NzlIMjQ2Ljc3MlY0MC4zOTdDMjQ2Ljc3MiAzOS4xMjc3IDI0Ni45NTIgMzguMTU4NyAyNDYuOTUyIDM4LjE1ODdIMjQ2LjgzNEMyNDQuOTc1IDQxLjQyODMgMjQxLjMxMiA0My43MjkgMjM3LjE3MyA0My43MjlDMjMxLjY1MSA0My43MjkgMjI3LjA5IDQwLjA5NjYgMjI3LjA5IDM0LjQ2NEMyMjcuMDgzIDI1LjE5MzMgMjM4LjEzMyAyMy42MTgxIDI0NC45MTMgMjMuNjE4MVpNMjM5LjI3MyAzNy44NDdDMjQzLjUzNiAzNy44NDcgMjQ2LjI5NCAzMy40ODkzIDI0Ni4yOTQgMjkuNzMyM1YyOC44ODI0SDI0NC45NzVDMjQxLjEzMiAyOC44ODI0IDIzNC42NSAyOS40MjY0IDIzNC42NSAzMy43ODk2QzIzNC42NSAzNS44NDY3IDIzNi4yMTEgMzcuODQ3IDIzOS4yNzMgMzcuODQ3WiIgZmlsbD0id2hpdGUiPjwvcGF0aD4KICAgIDxwYXRoIGQ9Ik0yNjMuODQ2IDQ5LjIzMUMyNjUuOTQ3IDQ5LjIzMSAyNjguMDQ5IDQ3Ljk2MTYgMjY5LjE4OSA0NS4yMzZMMjcwLjMyOSA0Mi41MTA0TDI1Ny40MjYgMTIuMTE0NEgyNjYuMDFMMjcyLjQzIDI5LjkxMzNDMjczLjAzIDMxLjU0NTMgMjczLjU3IDM0LjA4OTYgMjczLjU3IDM0LjA4OTZIMjczLjY5NEMyNzMuNjk0IDM0LjA4OTYgMjc0LjIzMiAzMS42NyAyNzQuNzE2IDMwLjAzMjNMMjgwLjcxNCAxMi4xMDg2SDI4OUwyNzUuMzczIDQ3LjY1NTdDMjczLjI3MiA1My4yODgzIDI2OC44MjkgNTUuNzA4IDI2NC4zMyA1NS43MDhDMjYwLjY2NyA1NS43MDggMjU3Ljk2NiA1My43MDc3IDI1Ny45NjYgNTMuNzA3N0wyNjAuNDg4IDQ3Ljk1NkMyNjAuNDg4IDQ3Ljk1NiAyNjIuMTY3IDQ5LjIzMSAyNjMuODQ2IDQ5LjIzMVoiIGZpbGw9IndoaXRlIj48L3BhdGg+CiAgICA8cGF0aCBkPSJNNjUuMTcwMyA0MC4xMjY0QzY1LjE3MDMgNDAuMDk4MSA2NS4xNzAzIDQwLjA2NDEgNjUuMTc1OCA0MC4wMzU4QzY1LjIxNTEgMzkuMDIxNCA2NC4zMTA3IDM4LjI0NTEgNjMuMzMzNCAzOC41MDAxQzYyLjc5NDEgMzguNjQxNyA2MS4zNjc0IDM5LjEwMDggNjEuMDMwNCAzOS4xMjkxQzYwLjY1NCAzOS4xNjMyIDYyLjc5NDEgMzIuMzk3MSA2My4zODM5IDI3LjM3NjRDNjMuNzc3MiAyNC4wNTU4IDYzLjQ5MDcgMjEuNTAwMSA2MS4zNTA1IDIyLjE5MTVDNTMuMTgzMyAyNy4wMDI1IDQ4LjExMSAzNC44NzM0IDQwLjU1NiA0MC4zMTM0QzQyLjgzNjYgMzAuOTQwOCA0Mi45NjAxIDI2LjM1MDggNDIuMTQwMSAyNS44ODYyQzQxLjIyNDUgMjUuMzcwNSAzOS4xMzQ5IDMwLjAxNzEgMzcuNTc4OSAzOC45MjUxQzM0Ljc0MjMgNTUuMTcxNSA1NS40MDc2IDI5Ljg2OTcgNTkuNzcyMiAyNy4wODE3QzU4LjU5ODIgMzMuNTg3MiA1OC40MjQgMzcuNjg5NyA1Ny4wNTkxIDQwLjE0OTFDNTYuNTc2IDQxLjAxNjEgNDkuODU4IDQ1LjAxNjggNDMuNDIwOCA1MC4zNjYxQzM4LjczNjEgNTQuMjY0OCAzMy40Nzg0IDYwLjk3OTggMzAuODA0NiA2Ny4xMTY4QzI4LjAyOTggNzMuNDgwNCAyNy45NDU2IDc5LjExODcgMzMuODIxIDc5LjkyOTFDMzYuNTExNiA4MC4yOTc0IDM5LjIxOTEgNzkuMTkyNSA0MS4yODA2IDc3LjQ2NDFDNTEuNTg4IDY4LjgxNjggNTYuODA2MyA1MS42OTc3IDU4Ljk1NzcgNDQuODYzOEM1OS4xNzEyIDQ0LjE4MzggNTkuMjY2NiA0My45NDAxIDU5LjM2MjEgNDMuNTg4OEM1OS40MjM5IDQzLjQ0MTQgNjIuMTk4OCA0Mi4xMjExIDY0LjExOTggNDEuNTMxOEM2NC43NDMzIDQxLjM1MDQgNjUuMTQ3OCA0MC43NzgxIDY1LjE3MDMgNDAuMTI2NFpNMzYuMDAwNSA3Ni45ODI0QzM1LjQ0NDQgNzcuMzkwNCAzNC43NzU5IDc3LjY5NjQgMzQuMDc5NCA3Ny43NjQ0QzI4LjYyNTIgNzguMzE0MSAzNi41MDA0IDY0LjAxNzEgMzcuNzgxMSA2MS43NDQ4QzQwLjQ2MDUgNTcuNzIxNCA0My42Njc5IDU0LjI0MjEgNDcuMDk5OSA1MS4yMDQ4QzQ4LjM2MzcgNTAuMDg4NCA1NS4xNjYxIDQ1LjMyMjggNTUuMTcxNyA0NS43OTMxQzQ5LjkxNDIgNTYuNjg0NSA0Ni4yMzQ5IDY5LjQwMDUgMzYuMDAwNSA3Ni45ODI0WiIgZmlsbD0iIzYxODZGQSI+PC9wYXRoPgogICAgPHBhdGggZD0iTTMwLjM3MTggNDQuMDU4OEMyOS41NDA1IDQ0Ljc5NTYgMjguMjMxNyA0NC4yMzQ1IDI4LjExMzcgNDMuMTIzOUMyNy45NzMzIDQxLjgzMTkgMjguMjE0OSA0MC4xNjAyIDI4LjY0NzMgMzguMzYzOUMzMC40NDQ5IDMwLjg4MzkgMzUuNTIyNyAyMS4xNzY5IDI5Ljg1NSAyNi45NTY5QzI3LjcwOTMgMjkuMTQ0MiAyMi4zOTU1IDM3LjQ4NTUgMjEuMTE0OCAzOS41MTQyQzE5LjY1NDQgNDEuODI2MiAxNy45OTE2IDQzLjQ4NjUgMTYuMTQ5MiA0MS4yODIyQzE3LjQwMTkgMzYuNzIwNSAxOS4zNzM1IDMyLjI0OTUgMTkuNTEzOSAyNi43MDc2QzE5LjU5MjUgMjMuNTM5OSA4LjA4MzA0IDM4LjU1NjUgNS40NTQyMiA0My4yMTQ1QzMuODE0MDMgNDYuMTIxNSAwLjAwNTYxNzEyIDQ1Ljg2MDkgMCA0My4xOTc1QzIuMTUxMzYgMzguMzY5NSAwLjcwNzc1NyAyNy4wODE1IDUuNzQwNyAyNS45MDI5QzYuMjU3NDggMjkuMDk4OSAzLjc4MDMyIDM4LjU1MDggMy44NTMzNCAzOC44MTcyQzMuOTcxMzEgMzkuMjU5MiA0LjcyOTYxIDM3LjkyMTggNS45ODc4NSAzNS43Njg1QzguNDU5MzkgMzAuNTgzNiAxNC41NDg0IDI0LjExNzggMTguNTE5NyAyMy4wNjM5QzIxLjYwOTEgMjIuMjQ3OSAyMy40MjM1IDI0LjcwNzIgMjEuMjYwOCAzMy41OTI1QzI0LjA5MTkgMjkuMzQyNSAzMC41MjM0IDIyLjU0MjYgMzMuOTg5MiAyMS45NTg5QzM1LjgwMzYgMjEuNjUyOSAzNi44MDM0IDIzLjA0NjkgMzYuMDYyIDI3LjM5MzJDMzUuODA5MSAyOC45MDYyIDMzLjUyMyA0MS4yODIyIDMwLjM3MTggNDQuMDU4OFoiIGZpbGw9IiM2MTg2RkEiPjwvcGF0aD4KICAgIDxwYXRoIGQ9Ik0xNjMuODczIDQwLjYyNDJWNDIuOTQ3NkgxODQuMzI1QzE4Ni43MjQgNDIuOTQ3NiAxODguNjY3IDQwLjk4NjkgMTg4LjY2NyAzOC41NjcyVjM2LjI0MzlIMTY4LjIxNkMxNjUuODE3IDM2LjI0MzkgMTYzLjg3MyAzOC4yMDQ1IDE2My44NzMgNDAuNjI0MloiIGZpbGw9IndoaXRlIj48L3BhdGg+CiAgICA8cGF0aCBkPSJNMTYzLjg3MyAxNi40NzNWMTguNzk2MkgxODQuMzI1QzE4Ni43MjQgMTguNzk2MiAxODguNjY3IDE2LjgzNTYgMTg4LjY2NyAxNC40MTU5VjEyLjA5MjVIMTY4LjIxNkMxNjUuODE3IDEyLjA5MjUgMTYzLjg3MyAxNC4wNTMzIDE2My44NzMgMTYuNDczWiIgZmlsbD0id2hpdGUiPjwvcGF0aD4KICAgIDxwYXRoIGQ9Ik0xNjMuODczIDI4LjU0ODhWMzAuODcyMUgxODQuMzI1QzE4Ni43MjQgMzAuODcyMSAxODguNjY3IDI4LjkxMTUgMTg4LjY2NyAyNi40OTE4VjI0LjE2ODVIMTY4LjIxNkMxNjUuODE3IDI0LjE2ODUgMTYzLjg3MyAyNi4xMjkxIDE2My44NzMgMjguNTQ4OFoiIGZpbGw9IndoaXRlIj48L3BhdGg+CiAgPC9nPgogIDxkZWZzPgogICAgPGNsaXBQYXRoIGlkPSJjbGlwMF8yNTRfNDQ0NyI+CiAgICAgIDxyZWN0IHdpZHRoPSIyODkiIGhlaWdodD0iODAiIGZpbGw9IndoaXRlIj48L3JlY3Q+CiAgICA8L2NsaXBQYXRoPgogIDwvZGVmcz4KPC9zdmc+Cg=="

# ────────────────────────────────────────────────────────────────────────
# Custom CSS — myBasePay Brand Guidelines
#
#   MBP Emerald #006633 (50%)  |  Prussian Blue #121631 (5%)
#   Wisteria Blue #7393f9 (10%)  |  Alabaster Gray #e8e8e8 (35%)
#   Marigold #ffbf00  |  Frosted Mint #e7fcdb  |  Icy Blue #b7d4f7
#   Font: Roboto  |  Corners: 12px
# ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Roboto from Google Fonts ───────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700;900&display=swap');
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif !important;
    }

    /* ── Sidebar — Prussian Blue ───────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #121631 0%, #0e1228 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e8e8e8 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-family: 'Roboto', sans-serif !important;
        font-weight: 900 !important;
    }
    [data-testid="stSidebar"] .stMarkdown a {
        color: #7393f9 !important;
    }

    /* ── Sidebar logo ──────────────────────────────────────────── */
    [data-testid="stSidebar"] .sidebar-logo {
        padding: 8px 0 4px 0;
    }
    [data-testid="stSidebar"] .sidebar-logo img {
        width: 180px;
        height: auto;
    }

    /* ── Sidebar app title ─────────────────────────────────────── */
    [data-testid="stSidebar"] .sidebar-title {
        font-family: 'Roboto', sans-serif !important;
        font-weight: 900 !important;
        font-size: 1.4rem;
        color: #ffffff !important;
        margin: 4px 0 2px 0;
    }

    /* ── Sidebar description ───────────────────────────────────── */
    [data-testid="stSidebar"] .sidebar-desc {
        font-size: 0.82rem;
        color: #afafaf !important;
        line-height: 1.5;
        margin-bottom: 0.5rem;
    }

    /* ── Sidebar source pills ──────────────────────────────────── */
    [data-testid="stSidebar"] .source-item {
        background: rgba(0, 102, 51, 0.12);
        border: 1px solid rgba(0, 102, 51, 0.3);
        border-radius: 12px;
        padding: 8px 12px;
        margin-bottom: 6px;
        font-size: 0.76rem;
        line-height: 1.4;
        color: #b7d4f7 !important;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    [data-testid="stSidebar"] .source-item .source-icon {
        margin-right: 6px;
    }

    /* ── Sidebar dividers ──────────────────────────────────────── */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.06) !important;
        margin: 12px 0 !important;
    }

    /* ── Clear Chat button ─────────────────────────────────────── */
    [data-testid="stSidebar"] button {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #afafaf !important;
        border-radius: 12px !important;
        font-family: 'Roboto', sans-serif !important;
        font-weight: 500 !important;
        transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] button:hover {
        background: rgba(255, 191, 0, 0.12) !important;
        border-color: rgba(255, 191, 0, 0.35) !important;
        color: #ffbf00 !important;
    }

    /* ── Chat input — MBP Emerald focus ────────────────────────── */
    .stChatInput > div {
        border-color: #e8e8e8 !important;
        border-radius: 12px !important;
    }
    .stChatInput > div:focus-within {
        border-color: #006633 !important;
        box-shadow: 0 0 0 1px #006633 !important;
    }

    /* ── Source expander ────────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-size: 0.85rem !important;
        color: #006633 !important;
        font-family: 'Roboto', sans-serif !important;
    }
    [data-testid="stExpander"] {
        border-radius: 12px !important;
        border-color: #e8e8e8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ────────────────────────────────────────────────────────────────────────
# Cached resource loaders
# ────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model …")
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    logger.info("Loading SentenceTransformer model …")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model loaded.")
    return model


@st.cache_resource(show_spinner="Indexing documents — this may take a minute on first boot …")
def build_index(_embed_model):
    logger.info(f"DOCUMENTS_DIR resolved to: {DOCUMENTS_DIR}")
    logger.info(f"  exists?  {os.path.isdir(DOCUMENTS_DIR)}")

    if os.path.isdir(DOCUMENTS_DIR):
        contents = os.listdir(DOCUMENTS_DIR)
        logger.info(f"  contents: {contents}")
    else:
        logger.error(f"documents/ NOT FOUND at {DOCUMENTS_DIR}")
        parent = os.path.dirname(DOCUMENTS_DIR)
        if os.path.isdir(parent):
            logger.error(f"  parent dir ({parent}) contains: {os.listdir(parent)}")
        return None, [], []

    try:
        chunks, filenames = ingest_all(DOCUMENTS_DIR)
    except Exception as e:
        logger.error(f"Ingestion crashed: {e}", exc_info=True)
        return None, [], []

    if not chunks:
        logger.warning("Ingestion returned 0 chunks.")
        return None, [], filenames

    try:
        index = build_vector_store(chunks, _embed_model)
    except Exception as e:
        logger.error(f"Embedding/FAISS crashed: {e}", exc_info=True)
        return None, [], filenames

    logger.info(f"Index ready: {index.ntotal} vectors from {filenames}")
    return index, chunks, filenames


def get_anthropic_client():
    api_key = st.secrets.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


# ────────────────────────────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────────────────────────────

def _pretty_filename(fn: str) -> tuple[str, str]:
    if fn.lower().endswith(".xlsx"):
        return "📊", fn.replace("_", " ").replace("-", " ").rsplit(".", 1)[0]
    else:
        return "📄", fn.replace("_", " ").replace("-", " ").rsplit(".", 1)[0]


# ────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # Logo
    st.markdown(
        f'''<div class="sidebar-logo">
            <img src="{LOGO_DATA_URI}" alt="myBasePay" />
        </div>''',
        unsafe_allow_html=True,
    )

    # App title + description
    st.markdown('<div class="sidebar-title">🎓 MBP University</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="sidebar-desc">{APP_DESCRIPTION}</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    # Load model & index
    embed_model = get_embedding_model()
    index, metadata, doc_filenames = build_index(embed_model)

    if doc_filenames:
        st.markdown("**📚 Loaded Sources**")
        for fn in doc_filenames:
            icon, display = _pretty_filename(fn)
            st.markdown(
                f'<div class="source-item">'
                f'<span class="source-icon">{icon}</span>{display}'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.warning("No documents found in the documents/ folder.")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ────────────────────────────────────────────────────────────────────────
# Validate essentials
# ────────────────────────────────────────────────────────────────────────

client = get_anthropic_client()
if client is None:
    st.error(
        "⚠️ **Anthropic API key not found.** "
        "Add `ANTHROPIC_API_KEY` to your Streamlit Cloud Secrets "
        "(Settings → Secrets)."
    )
    st.stop()

if index is None or not metadata:
    st.error(
        "📭 **No documents indexed.**\n\n"
        f"- **Documents dir checked:** `{DOCUMENTS_DIR}`\n"
        f"- **Directory exists:** `{os.path.isdir(DOCUMENTS_DIR)}`\n"
        f"- **Contents:** `{os.listdir(DOCUMENTS_DIR) if os.path.isdir(DOCUMENTS_DIR) else 'N/A'}`\n\n"
        "Make sure `.docx` and `.xlsx` files are **committed** inside the "
        "`documents/` folder in your GitHub repo, then reboot the app."
    )
    st.stop()


# ────────────────────────────────────────────────────────────────────────
# Chat interface
# ────────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🎓"):
        st.markdown(WELCOME_MESSAGE)

for msg in st.session_state.messages:
    avatar = "🎓" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 View Sources"):
                st.markdown(msg["sources"])

if prompt := st.chat_input("Ask MBP University a question …"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    results = search(prompt, index, metadata, embed_model)
    context_block = format_context(results)
    sources_display = format_sources_for_display(results)

    claude_messages: list[dict] = []
    for m in st.session_state.messages[-11:-1]:
        claude_messages.append({"role": m["role"], "content": m["content"]})

    user_content = (
        f"## Retrieved Document Context\n\n{context_block}\n\n---\n\n"
        f"## User Question\n\n{prompt}"
    )
    claude_messages.append({"role": "user", "content": user_content})

    with st.chat_message("assistant", avatar="🎓"):
        placeholder = st.empty()
        full_response = ""

        try:
            with client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=claude_messages,
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

            with st.expander("📄 View Sources"):
                st.markdown(sources_display)

        except anthropic.APIError as e:
            full_response = f"⚠️ API error: {e.message}"
            placeholder.markdown(full_response)
            sources_display = ""
        except Exception as e:
            full_response = f"⚠️ Unexpected error: {e}"
            placeholder.markdown(full_response)
            sources_display = ""

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources_display,
    })
