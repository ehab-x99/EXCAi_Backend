import streamlit as st
import io
import os
import asyncio
import tempfile as temp
from dotenv import load_dotenv
import sys
import time
import streamlit.components.v1 as components

from pathlib import Path
from audio import PODCAST_GEN
from typing import Tuple
from workflow import NotebookLMWorkflow, FileInputEvent, NotebookOutputEvent
from instrumentation import OtelTracesSqlEngine
from llama_index.observability.otel import LlamaIndexOpenTelemetry
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)

load_dotenv()

# define a custom span exporter
span_exporter = OTLPSpanExporter("http://localhost:4318/v1/traces")

# initialize the instrumentation object
instrumentor = LlamaIndexOpenTelemetry(
    service_name_or_resource="agent.traces",
    span_exporter=span_exporter,
    debug=True,
)
sql_engine = OtelTracesSqlEngine(
    engine_url=f"postgresql+psycopg2://{os.getenv('pgql_user')}:{os.getenv('pgql_psw')}@localhost:5432/{os.getenv('pgql_db')}",
    table_name="agent_traces",
    service_name="agent.traces",
)

WF = NotebookLMWorkflow(timeout=600)


# Read the HTML file
def read_html_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


async def run_workflow(file: io.BytesIO) -> Tuple[str, str, str, str, str]:
    # Create temp file with proper Windows handling
    with temp.NamedTemporaryFile(suffix=".pdf", delete=False) as fl:
        content = file.getvalue()
        fl.write(content)
        fl.flush()  # Ensure data is written
        temp_path = fl.name

    try:
        st_time = int(time.time() * 1000000)
        ev = FileInputEvent(file=temp_path)
        result: NotebookOutputEvent = await WF.run(start_event=ev)

        q_and_a = ""
        for q, a in zip(result.questions, result.answers):
            q_and_a += f"**{q}**\n\n{a}\n\n"
        bullet_points = "## Bullet Points\n\n- " + "\n- ".join(result.highlights)

        mind_map = result.mind_map
        if Path(mind_map).is_file():
            mind_map = read_html_file(mind_map)
            try:
                os.remove(result.mind_map)
            except OSError:
                pass  # File might be locked on Windows

        end_time = int(time.time() * 1000000)
        sql_engine.to_sql_database(start_time=st_time, end_time=end_time)
        return result.md_content, result.summary, q_and_a, bullet_points, mind_map

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            await asyncio.sleep(0.1)
            try:
                os.remove(temp_path)
            except OSError:
                pass  # Give up if still locked


def sync_run_workflow(file: io.BytesIO):
    try:
        # Try to use existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, schedule the coroutine
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, run_workflow(file))
                return future.result()
        else:
            return loop.run_until_complete(run_workflow(file))
    except RuntimeError:
        # No event loop exists, create one
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        return asyncio.run(run_workflow(file))


async def create_podcast(file_content: str):
    audio_fl = await PODCAST_GEN.create_conversation(file_transcript=file_content)
    return audio_fl


def sync_create_podcast(file_content: str):
    return asyncio.run(create_podcast(file_content=file_content))


# Display the network
st.set_page_config(
    page_title="EXCAi_3.0 - Home",
    page_icon="🏠",
    layout="wide",
    menu_items={
        "Get Help": "https://github.com/run-llama/notebooklm-clone/discussions/categories/general",
        "Report a bug": "https://github.com/run-llama/notebooklm-clone/issues/",
        "About": "An OSS alternative to NotebookLM that runs with the power of a flully Llama!",
    },
)
st.sidebar.header("Home🏠")
st.sidebar.info("To switch to the Document Chat, select it from above!🔺")
st.markdown("---")
st.markdown("## EXCAi_3.0 - Home🦙")

file_input = st.file_uploader(
    label="Upload your source PDF file!", accept_multiple_files=False
)

# Add this after your existing code, before the st.title line:

# Initialize session state
if "workflow_results" not in st.session_state:
    st.session_state.workflow_results = None

if file_input is not None:
    # First button: Process Document
    if st.button("Process Document", type="primary"):
        with st.spinner("Processing document... This may take a few minutes."):
            try:
                md_content, summary, q_and_a, bullet_points, mind_map = (
                    sync_run_workflow(file_input)
                )
                st.session_state.workflow_results = {
                    "md_content": md_content,
                    "summary": summary,
                    "q_and_a": q_and_a,
                    "bullet_points": bullet_points,
                    "mind_map": mind_map,
                }
                st.success("Document processed successfully!")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

    # Display results if available
    if st.session_state.workflow_results:
        results = st.session_state.workflow_results

        # Summary
        st.markdown("## Summary")
        st.markdown(results["summary"])

        # Bullet Points
        st.markdown(results["bullet_points"])

        # FAQ (toggled)
        with st.expander("FAQ"):
            st.markdown(results["q_and_a"])

        # Mind Map
        if results["mind_map"]:
            st.markdown("## Mind Map")
            components.html(results["mind_map"], height=800, scrolling=True)

        # Second button: Generate Podcast
        if st.button("Generate In-Depth Conversation", type="secondary"):
            with st.spinner("Generating podcast... This may take several minutes."):
                try:
                    audio_file = sync_create_podcast(results["md_content"])
                    st.success("Podcast generated successfully!")

                    # Display audio player
                    st.markdown("## Generated Podcast")
                    if os.path.exists(audio_file):
                        with open(audio_file, "rb") as f:
                            audio_bytes = f.read()
                        os.remove(audio_file)
                        st.audio(audio_bytes, format="audio/mp3")
                    else:
                        st.error("Audio file not found.")

                except Exception as e:
                    st.error(f"Error generating podcast: {str(e)}")

else:
    st.info("Please upload a PDF file to get started.")

if __name__ == "__main__":
    instrumentor.start_registering()
