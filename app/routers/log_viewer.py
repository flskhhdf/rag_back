"""
Real-time log viewer API endpoints.
Provides endpoints to view and stream log files in real-time.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pathlib import Path
import asyncio
from datetime import datetime

router = APIRouter(prefix="/logs", tags=["logs"])

# Log file path
LOG_DIR = Path("logs")


@router.get("/")
async def get_log_viewer():
    """Serve the log viewer HTML page."""
    html_path = Path(__file__).parent.parent / "static" / "log_viewer.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Log viewer page not found")
    return FileResponse(html_path)


@router.get("/stream")
async def stream_logs():
    """Stream log file in real-time using Server-Sent Events."""
    
    async def generate():
        # Get today's log file
        today = datetime.now().strftime("%Y_%m_%d")
        log_file = LOG_DIR / today / "rag.log"
        
        if not log_file.exists():
            yield f"data: Log file not found: {log_file}\n\n"
            return
        
        # Read existing content first
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                content = f.read()
                if content:
                    # Send existing content
                    for line in content.splitlines():
                        yield f"data: {line}\n\n"
        except Exception as e:
            yield f"data: Error reading log file: {str(e)}\n\n"
            return
        
        # Follow the file for new lines (like tail -f)
        last_position = log_file.stat().st_size
        
        while True:
            try:
                current_size = log_file.stat().st_size
                
                # If file has grown, read new content
                if current_size > last_position:
                    with open(log_file, "r", encoding="utf-8") as f:
                        f.seek(last_position)
                        new_lines = f.read()
                        
                        for line in new_lines.splitlines():
                            if line.strip():
                                yield f"data: {line}\n\n"
                        
                        last_position = current_size
                
                # Wait before checking again
                await asyncio.sleep(0.5)
                
            except Exception as e:
                yield f"data: Error: {str(e)}\n\n"
                await asyncio.sleep(1)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@router.get("/list")
async def list_log_files():
    """List available log files."""
    if not LOG_DIR.exists():
        return {"files": []}
    
    log_files = []
    for date_dir in sorted(LOG_DIR.iterdir(), reverse=True):
        if date_dir.is_dir():
            for log_file in date_dir.glob("*.log"):
                log_files.append({
                    "name": f"{date_dir.name}/{log_file.name}",
                    "size": log_file.stat().st_size,
                    "modified": log_file.stat().st_mtime
                })
    
    return {"files": log_files}
