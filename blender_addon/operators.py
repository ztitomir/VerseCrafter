"""
VerseCrafter Blender Addon Operators

Defines operators for workflow actions.
"""

import bpy
import os
import json
import subprocess
import math
from bpy.types import Operator
from bpy.props import StringProperty

# Add user site-packages to path (for macOS where Blender's site-packages is not writable)
import sys
import site

# Try to add user site-packages
user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.insert(0, user_site)

# Also try common user pip install locations
import os
home = os.path.expanduser("~")
possible_paths = [
    os.path.join(home, ".local", "lib", "python3.11", "site-packages"),
    os.path.join(home, "Library", "Python", "3.11", "lib", "python", "site-packages"),
]
for p in possible_paths:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

# Check dependencies individually for better error reporting
HAS_DEPS = True
MISSING_DEPS = []

try:
    import numpy as np
except ImportError:
    HAS_DEPS = False
    MISSING_DEPS.append("numpy")

try:
    import cv2
except ImportError:
    HAS_DEPS = False
    MISSING_DEPS.append("opencv-python (cv2)")

try:
    import mathutils
except ImportError:
    # mathutils is built into Blender, this shouldn't fail
    pass

try:
    import matplotlib
except ImportError:
    HAS_DEPS = False
    MISSING_DEPS.append("matplotlib")

if not HAS_DEPS:
    print(f"Warning: Missing dependencies: {', '.join(MISSING_DEPS)}. Some features may not work.")
    print("Install with: pip install " + " ".join([d.split()[0] for d in MISSING_DEPS]))


# ============================================================================
# File Transfer Utilities
# ============================================================================

import urllib.request
import urllib.parse
import urllib.error
import base64
import tempfile
import time


def test_server_connection(server_url: str, cookie: str = None) -> tuple:
    """
    Test connection to the server.
    
    Args:
        server_url: The server URL to connect to
        cookie: Optional cookie string for proxy authentication
    
    Returns:
        Tuple of (success, message)
    """
    import ssl
    global _proxy_cookie
    
    try:
        url = f"{server_url}/health"
        req = urllib.request.Request(url)
        req.add_header('Accept', 'application/json')
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        req.add_header('Content-Type', 'application/json')
        
        # Add cookie if provided or if global cookie is set
        use_cookie = cookie if cookie else _proxy_cookie
        if use_cookie:
            req.add_header('Cookie', use_cookie)
            print(f"Using cookie: {use_cookie[:50]}..." if len(use_cookie) > 50 else f"Using cookie: {use_cookie}")
        
        # Create SSL context that doesn't verify certificates (for proxies)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(req, timeout=15, context=ssl_context) as response:
            response_text = response.read().decode('utf-8')
            
            # Debug: print first 500 chars of response
            print(f"Server URL: {url}")
            print(f"Server response status: {response.status}")
            print(f"Server response headers: {dict(response.headers)}")
            print(f"Server response body: {response_text[:500]}")
            
            # Check if response looks like HTML (proxy error page)
            if response_text.strip().startswith('<') or 'html' in response_text.lower()[:100]:
                # Try to extract error message from HTML
                if 'login' in response_text.lower() or 'sign in' in response_text.lower():
                    return False, f"Proxy requires authentication. Set 'Proxy Cookie' from browser."
                elif '403' in response_text or 'forbidden' in response_text.lower():
                    return False, f"Proxy returned 403 Forbidden. Access denied."
                elif '502' in response_text or 'bad gateway' in response_text.lower():
                    return False, f"Proxy returned 502 Bad Gateway. Server may not be running."
                else:
                    return False, f"Proxy returned HTML instead of JSON. Open in browser to check: {url}"
            
            try:
                result = json.loads(response_text)
                if result.get('status') == 'ok':
                    return True, "Connection successful!"
                else:
                    return False, f"Server responded: {result}"
            except json.JSONDecodeError:
                return False, f"Invalid JSON response: {response_text[:100]}..."
                
    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode('utf-8')[:200]
        except:
            pass
        return False, f"HTTP Error {e.code}: {e.reason}. Body: {error_body}"
    except urllib.error.URLError as e:
        return False, f"URL Error: {e.reason}"
    except ssl.SSLError as e:
        return False, f"SSL Error: {str(e)}. Try using http:// instead of https://"
    except Exception as e:
        import traceback
        print(f"Connection error traceback: {traceback.format_exc()}")
        return False, f"Error: {str(e)}"


def _get_ssl_context():
    """Create SSL context that doesn't verify certificates (for proxies)."""
    import ssl
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    return ssl_context


# Global cookie storage (set from Blender properties)
_proxy_cookie = ""

def set_proxy_cookie(cookie: str):
    """Set the proxy cookie for all requests."""
    global _proxy_cookie
    _proxy_cookie = cookie
    print(f"Proxy cookie set: {cookie[:50]}..." if len(cookie) > 50 else f"Proxy cookie set: {cookie}")


def _make_request(url: str, timeout: int = 60, cookie: str = None):
    """Make HTTP request with proper headers, SSL handling, and optional cookie."""
    global _proxy_cookie
    req = urllib.request.Request(url)
    req.add_header('Accept', 'application/json')
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    # Use provided cookie or global cookie
    use_cookie = cookie if cookie else _proxy_cookie
    if use_cookie:
        req.add_header('Cookie', use_cookie)
    
    return urllib.request.urlopen(req, timeout=timeout, context=_get_ssl_context())


def upload_file_to_server(server_url: str, local_path: str, target_dir: str) -> tuple:
    """
    Upload a file to the server using POST request with multipart form data.
    Falls back to base64 GET for small files if POST fails.
    
    Returns:
        Tuple of (success, server_path_or_error)
    """
    global _proxy_cookie
    
    try:
        with open(local_path, 'rb') as f:
            file_data = f.read()
        
        filename = os.path.basename(local_path)
        file_size = len(file_data)
        
        print(f"Uploading file: {filename} ({file_size} bytes)")
        
        # Try POST request first (works with cookie auth)
        try:
            return _upload_file_post(server_url, filename, file_data, target_dir)
        except Exception as e:
            print(f"POST upload failed: {e}, trying base64 method...")
        
        # Fallback: For small files, use base64 GET request
        if file_size <= 10000:  # ~10KB limit for URL
            data_b64 = base64.b64encode(file_data).decode('utf-8')
            params = urllib.parse.urlencode({
                'filename': filename,
                'target_dir': target_dir,
                'data': data_b64
            })
            
            url = f"{server_url}/api/upload_base64?{params}"
            
            with _make_request(url, timeout=60) as response:
                response_text = response.read().decode('utf-8')
                if not response_text:
                    return False, "Empty response from server"
                    
                if response_text.strip().startswith('<'):
                    return False, "Server returned HTML (proxy blocking). Check browser auth."
                    
                result = json.loads(response_text)
                if result.get('success'):
                    return True, result.get('path')
                else:
                    return False, result.get('error', 'Unknown error')
        else:
            return False, f"File too large ({file_size} bytes). Use 'Server Image Path' instead."
                    
    except json.JSONDecodeError as e:
        return False, f"Server returned invalid JSON (proxy might be blocking)"
    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode('utf-8')[:100]
        except:
            pass
        return False, f"HTTP Error {e.code}: {e.reason}. {error_body}"
    except urllib.error.URLError as e:
        return False, f"URL Error: {e.reason}"
    except Exception as e:
        import traceback
        print(f"Upload error: {traceback.format_exc()}")
        return False, str(e)


def _upload_file_post(server_url: str, filename: str, file_data: bytes, target_dir: str) -> tuple:
    """Upload file using POST request with multipart form data."""
    global _proxy_cookie
    import uuid as uuid_module
    
    boundary = f'----WebKitFormBoundary{uuid_module.uuid4().hex}'
    
    # Build multipart form data
    body_parts = []
    
    # File part
    body_parts.append(f'--{boundary}'.encode())
    body_parts.append(f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode())
    body_parts.append(b'Content-Type: application/octet-stream')
    body_parts.append(b'')
    body_parts.append(file_data)
    
    # Target directory part
    body_parts.append(f'--{boundary}'.encode())
    body_parts.append(b'Content-Disposition: form-data; name="target_dir"')
    body_parts.append(b'')
    body_parts.append(target_dir.encode())
    
    # End boundary
    body_parts.append(f'--{boundary}--'.encode())
    
    body = b'\r\n'.join(body_parts)
    
    url = f"{server_url}/api/upload"
    req = urllib.request.Request(url, data=body, method='POST')
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    req.add_header('Accept', 'application/json')
    
    if _proxy_cookie:
        req.add_header('Cookie', _proxy_cookie)
    
    with urllib.request.urlopen(req, timeout=120, context=_get_ssl_context()) as response:
        response_text = response.read().decode('utf-8')
        
        if response_text.strip().startswith('<'):
            raise Exception("Server returned HTML (proxy blocking POST)")
        
        result = json.loads(response_text)
        if result.get('success'):
            return True, result.get('path')
        else:
            return False, result.get('error', 'Unknown error')


def download_file_from_server(server_url: str, server_path: str, local_path: str) -> tuple:
    """
    Download a file from the server using base64 encoding (GET request compatible).
    
    Returns:
        Tuple of (success, local_path_or_error)
    """
    try:
        params = urllib.parse.urlencode({'path': server_path})
        url = f"{server_url}/api/download_base64?{params}"
        
        with _make_request(url, timeout=120) as response:
            response_text = response.read().decode('utf-8')
            
            # Check for HTML response
            if response_text.strip().startswith('<'):
                return False, "Server returned HTML (proxy blocking). Check browser auth."
            
            result = json.loads(response_text)
            
            if result.get('success'):
                data_b64 = result.get('data')
                file_data = base64.b64decode(data_b64)
                
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, 'wb') as f:
                    f.write(file_data)
                
                return True, local_path
            else:
                return False, result.get('error', 'Unknown error')
                
    except json.JSONDecodeError as e:
        return False, f"Server returned invalid JSON (proxy might be blocking)"
    except Exception as e:
        import traceback
        print(f"Download error: {traceback.format_exc()}")
        return False, str(e)


def check_task_status(server_url: str, task_id: str) -> dict:
    """Check task status from server."""
    try:
        url = f"{server_url}/api/status/{task_id}"
        with _make_request(url, timeout=30) as response:
            response_text = response.read().decode('utf-8')
            
            # Check for HTML response
            if response_text.strip().startswith('<'):
                return {"status": "error", "error": "Proxy returned HTML. Auth required."}
            
            return json.loads(response_text)
    except json.JSONDecodeError:
        return {"status": "error", "error": "Invalid JSON (proxy blocking)"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def start_preprocess_on_server(server_url: str, image_path: str, output_dir: str, text_prompt: str) -> tuple:
    """Start preprocessing on server via GET request."""
    try:
        params = urllib.parse.urlencode({
            'image_path': image_path,
            'output_dir': output_dir,
            'text_prompt': text_prompt
        })
        url = f"{server_url}/api/preprocess_get?{params}"
        
        print(f"Starting preprocess: {url}")
        
        with _make_request(url, timeout=30) as response:
            response_text = response.read().decode('utf-8')
            print(f"Preprocess response: {response_text[:500]}")
            
            # Check for HTML response
            if response_text.strip().startswith('<'):
                return False, "Server returned HTML (proxy blocking). Login in browser first."
            
            result = json.loads(response_text)
            if 'task_id' in result:
                return True, result['task_id']
            else:
                return False, result.get('error', 'No task_id returned')
    except json.JSONDecodeError:
        return False, "Server returned invalid JSON (proxy might be blocking)"
    except Exception as e:
        import traceback
        print(f"Preprocess start error: {traceback.format_exc()}")
        return False, str(e)


class VERSECRAFTER_OT_preprocess(Operator):
    """Run VerseCrafter preprocessing (Steps 1-3) on GPU server with auto file transfer"""
    bl_idname = "versecrafter.preprocess"
    bl_label = "Run Preprocess"
    bl_description = "Upload image, run preprocessing on GPU server, download results"
    
    _timer = None
    _task_id = None
    _server_output_dir = None
    _poll_count = 0  # Track polling count for adaptive interval
    
    def execute(self, context):
        props = context.scene.versecrafter
        
        # Validate inputs
        if not props.input_image_path:
            self.report({'ERROR'}, "Please select an input image")
            return {'CANCELLED'}
        
        if not os.path.exists(props.input_image_path):
            self.report({'ERROR'}, f"Image not found: {props.input_image_path}")
            return {'CANCELLED'}
        
        if not props.workflow_dir:
            self.report({'ERROR'}, "Please set a local workflow directory")
            return {'CANCELLED'}
        
        if not props.server_url:
            self.report({'ERROR'}, "Please set the server URL")
            return {'CANCELLED'}
        
        # Set global cookie if provided
        if props.proxy_cookie:
            set_proxy_cookie(props.proxy_cookie)
        
        # Create local workflow directory
        os.makedirs(props.workflow_dir, exist_ok=True)
        
        server_url = props.server_url.rstrip('/')
        
        # Reset all progress states when starting new preprocessing
        props.step1_complete = False
        props.step2_complete = False
        props.step3_complete = False
        props.step4_complete = False
        props.step5_complete = False
        props.step6_complete = False
        
        props.is_processing = True
        
        # Upload image to server
        self.report({'INFO'}, "Uploading image to server...")
        props.status_message = "Uploading image..."
        image_filename = os.path.basename(props.input_image_path)
        
        # Define server output directory (relative path, server will resolve to absolute)
        server_output_dir = f"outputs/{os.path.splitext(image_filename)[0]}_{int(time.time())}"
        
        success, result = upload_file_to_server(
            server_url,
            props.input_image_path,
            server_output_dir
        )
        
        if not success:
            props.is_processing = False
            self.report({'ERROR'}, f"Failed to upload image: {result}")
            props.status_message = f"✗ Upload failed: {result}"
            return {'CANCELLED'}
        
        server_image_path = result
        self.report({'INFO'}, f"Image uploaded to: {server_image_path}")
        
        VERSECRAFTER_OT_preprocess._server_output_dir = server_output_dir
        
        # Step 2: Start preprocessing on server
        props.status_message = "Starting preprocessing on server..."
        
        success, task_id = start_preprocess_on_server(
            server_url,
            server_image_path,
            server_output_dir,
            props.text_prompt
        )
        
        if not success:
            props.is_processing = False
            self.report({'ERROR'}, f"Failed to start preprocessing: {task_id}")
            return {'CANCELLED'}
        
        VERSECRAFTER_OT_preprocess._task_id = task_id
        self.report({'INFO'}, f"Preprocessing started, task ID: {task_id}")
        props.status_message = f"Processing... (Task: {task_id[:8]}...)"
        
        # Register timer to check completion
        VERSECRAFTER_OT_preprocess._poll_count = 0
        VERSECRAFTER_OT_preprocess._timer = bpy.app.timers.register(
            self._check_preprocess_completion, 
            first_interval=3.0
        )
        
        return {'FINISHED'}
    
    @staticmethod
    def _check_preprocess_completion():
        """Timer callback to check preprocessing status and download results."""
        props = bpy.context.scene.versecrafter
        server_url = props.server_url.rstrip('/')
        task_id = VERSECRAFTER_OT_preprocess._task_id
        server_output_dir = VERSECRAFTER_OT_preprocess._server_output_dir
        
        if not task_id:
            props.is_processing = False
            return None
        
        # Increment poll count for adaptive interval
        VERSECRAFTER_OT_preprocess._poll_count += 1
        
        # Check task status
        status = check_task_status(server_url, task_id)
        
        if status.get('status') == 'running':
            progress = status.get('progress', 0) * 100
            message = status.get('message', 'Processing...')
            props.status_message = f"{message} ({progress:.0f}%)"
            # Adaptive polling: start at 3s, increase to max 10s after 10 polls
            poll_count = VERSECRAFTER_OT_preprocess._poll_count
            interval = min(3.0 + poll_count * 0.5, 10.0)
            return interval
        
        elif status.get('status') == 'completed':
            props.status_message = "Downloading results..."
            
            # Download result files
            local_dir = props.workflow_dir
            
            files_to_download = [
                (f"{server_output_dir}/estimated_depth/depth_intrinsics.npz", 
                 os.path.join(local_dir, "estimated_depth", "depth_intrinsics.npz")),
                (f"{server_output_dir}/fitted_3D_gaussian/gaussian_params.json",
                 os.path.join(local_dir, "fitted_3D_gaussian", "gaussian_params.json")),
            ]
            
            # Also download the input image
            result_data = status.get('result', {})
            if result_data:
                # Try to get input image path from result
                pass
            
            download_success = True
            for server_path, local_path in files_to_download:
                success, result = download_file_from_server(server_url, server_path, local_path)
                if not success:
                    print(f"Failed to download {server_path}: {result}")
                    download_success = False
            
            # Copy local input image to workflow dir
            if props.input_image_path and os.path.exists(props.input_image_path):
                import shutil
                dest_path = os.path.join(local_dir, os.path.basename(props.input_image_path))
                if not os.path.exists(dest_path):
                    shutil.copy2(props.input_image_path, dest_path)
            
            if download_success:
                props.step1_complete = True
                props.step2_complete = True
                props.step3_complete = True
                props.status_message = "✓ Preprocessing complete! Click 'Load Scene Data' to continue."
                
                # Auto-load scene
                try:
                    bpy.ops.versecrafter.load_scene()
                except:
                    pass
            else:
                props.status_message = "⚠ Preprocessing done but some downloads failed. Check console."
            
            props.is_processing = False
            VERSECRAFTER_OT_preprocess._task_id = None
            return None
        
        elif status.get('status') == 'failed':
            props.status_message = f"✗ Failed: {status.get('error', 'Unknown error')}"
            props.is_processing = False
            VERSECRAFTER_OT_preprocess._task_id = None
            return None
        
        else:
            # Unknown status, keep checking with adaptive interval
            poll_count = VERSECRAFTER_OT_preprocess._poll_count
            interval = min(3.0 + poll_count * 0.5, 10.0)
            return interval


class VERSECRAFTER_OT_load_scene(Operator):
    """Load preprocessed data into Blender scene"""
    bl_idname = "versecrafter.load_scene"
    bl_label = "Load Scene Data"
    bl_description = "Load depth, masks, and 3D Gaussians into Blender"
    
    # Configuration parameters (matching build_4d_control_scene.py)
    PIXEL_STEP = 1
    ELLIPSOID_SCALE_FACTOR = 2.0
    ELLIPSOID_SEGMENTS = 128
    ELLIPSOID_RINGS = 64
    SNAPSHOT_OPACITY = 0.6
    
    def execute(self, context):
        props = context.scene.versecrafter
        
        if not props.workflow_dir:
            self.report({'ERROR'}, "Please set a workflow directory (local path to downloaded files)")
            return {'CANCELLED'}
        
        # Check for required files
        workflow_dir = props.workflow_dir
        
        # Try to find input image
        input_image = props.input_image_path
        if not input_image or not os.path.exists(input_image):
            # Try common locations
            for img_name in ["0001.png", "input.png", "input_image.png"]:
                test_path = os.path.join(workflow_dir, img_name)
                if os.path.exists(test_path):
                    input_image = test_path
                    break
        
        if not input_image or not os.path.exists(input_image):
            self.report({'ERROR'}, "Input image not found. Please set 'Input Image' path.")
            return {'CANCELLED'}
        
        depth_npz = os.path.join(workflow_dir, "estimated_depth", "depth_intrinsics.npz")
        gaussian_json = os.path.join(workflow_dir, "fitted_3D_gaussian", "gaussian_params.json")
        
        if not os.path.exists(depth_npz):
            self.report({'ERROR'}, f"Depth data not found at: {depth_npz}")
            return {'CANCELLED'}
        
        if not os.path.exists(gaussian_json):
            self.report({'ERROR'}, f"Gaussian data not found at: {gaussian_json}")
            return {'CANCELLED'}
        
        try:
            # Execute the scene building logic directly
            self.report({'INFO'}, "Loading scene data...")
            
            if not HAS_DEPS:
                missing = ", ".join(MISSING_DEPS) if MISSING_DEPS else "numpy, cv2, matplotlib"
                self.report({'ERROR'}, f"Missing dependencies: {missing}. Install them in Blender's Python.")
                return {'CANCELLED'}
            
            # ================== Cleanup Old Objects ==================
            self._cleanup_old_objects()
            
            # ================== Load NPZ Data ==================
            data = np.load(depth_npz)
            depths = data["depth"]
            intrinsics = data["intrinsic"]
            
            if depths.ndim == 2:
                depths = depths[np.newaxis, ...]
            
            height_d, width_d = depths.shape[1], depths.shape[2]
            
            # Denormalize intrinsics if needed
            if intrinsics.ndim == 2:
                if intrinsics[0, 0] < 10 or intrinsics[1, 1] < 10:
                    print(f"[INFO] Detected normalized intrinsics, converting to pixel coordinates...")
                    intrinsics = intrinsics.copy()
                    intrinsics[0, 0] *= width_d
                    intrinsics[1, 1] *= height_d
                    intrinsics[0, 2] *= width_d
                    intrinsics[1, 2] *= height_d
                    print(f"[INFO] Denormalized K: fx={intrinsics[0,0]:.2f}, fy={intrinsics[1,1]:.2f}")
            elif intrinsics.ndim == 3:
                if intrinsics[0, 0, 0] < 10 or intrinsics[0, 1, 1] < 10:
                    print(f"[INFO] Detected normalized intrinsics, converting to pixel coordinates...")
                    intrinsics = intrinsics.copy()
                    intrinsics[:, 0, 0] *= width_d
                    intrinsics[:, 1, 1] *= height_d
                    intrinsics[:, 0, 2] *= width_d
                    intrinsics[:, 1, 2] *= height_d
            
            # ================== Coordinate Transform Matrix ==================
            # OpenCV: X=right, Y=down, Z=forward
            # Blender (Z-up): X=right, Y=forward, Z=up
            COORD_TRANSFORM = np.array([
                [1,  0,  0],
                [0,  0,  1],
                [0, -1,  0],
            ], dtype=np.float32)
            
            # ================== Load PNG Image ==================
            rgb_image = cv2.imread(input_image)
            if rgb_image is None:
                self.report({'ERROR'}, f"Cannot open image: {input_image}")
                return {'CANCELLED'}
            
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            height_v, width_v = rgb_image.shape[:2]
            print(f"[INFO] Loaded PNG image: {height_v}x{width_v}")
            
            # ================== Resolution Alignment ==================
            if (height_d, width_d) != (height_v, width_v):
                print("[WARN] Depth vs RGB size mismatch, resizing depth to RGB size.")
                depth = cv2.resize(depths[0], (width_v, height_v), interpolation=cv2.INTER_NEAREST)
            else:
                depth = depths[0]
            
            depth = depth.astype(np.float32)
            print(f"[INFO] Depth range: min={depth.min():.3f}, max={depth.max():.3f}, mean={depth.mean():.3f}")
            
            K = intrinsics if intrinsics.ndim == 2 else intrinsics[0]
            K = K.astype(np.float32)
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            # ================== Create Background Point Cloud ==================
            print("[INFO] Creating point cloud from first frame...")
            background_verts = []
            background_colors = []
            
            for v in range(0, height_v, self.PIXEL_STEP):
                for u in range(0, width_v, self.PIXEL_STEP):
                    d_val = depth[v, u]
                    if d_val <= 0 or not np.isfinite(d_val):
                        continue
                    
                    # Back-project to camera coordinates (OpenCV convention)
                    x_cv = (u - cx) / fx * d_val
                    y_cv = (v - cy) / fy * d_val
                    z_cv = d_val
                    P_cv_opencv = np.array([x_cv, y_cv, z_cv], dtype=np.float32)
                    
                    # Convert camera coordinates from OpenCV to Blender
                    P_cv_blender = COORD_TRANSFORM @ P_cv_opencv
                    
                    background_verts.append((float(P_cv_blender[0]), float(P_cv_blender[1]), float(P_cv_blender[2])))
                    color = rgb_image[v, u]
                    background_colors.append((
                        float(color[0]) / 255.0,
                        float(color[1]) / 255.0,
                        float(color[2]) / 255.0,
                        1.0
                    ))
            
            print(f"[INFO] Point count: {len(background_verts)}")
            self.report({'INFO'}, f"Created {len(background_verts)} points")
            
            # ================== Create Point Cloud Mesh ==================
            zs = np.array([p[2] for p in background_verts], dtype=np.float32)
            zs_abs = np.abs(zs[np.isfinite(zs)])
            median_z = float(np.median(zs_abs)) if zs_abs.size > 0 else 1.0
            print(f"[INFO] Median depth: {median_z:.3f}")
            
            uniform_point_size = median_z * 0.01
            print(f"[INFO] Uniform point size: {uniform_point_size:.6f}")
            
            mesh_verts = []
            mesh_faces = []
            mesh_colors = []
            
            for idx, (x, y, z) in enumerate(background_verts):
                r, g, b, a = background_colors[idx]
                size = uniform_point_size
                
                base = len(mesh_verts)
                mesh_verts.extend([
                    (x - size, y, z - size),
                    (x + size, y, z - size),
                    (x + size, y, z + size),
                    (x - size, y, z + size),
                ])
                mesh_faces.append((base, base + 1, base + 2, base + 3))
                mesh_colors.extend([(r, g, b, a)] * 4)
            
            mesh = bpy.data.meshes.new("PointCloudMesh")
            mesh.from_pydata(mesh_verts, [], mesh_faces)
            mesh.update()
            
            pc_obj = bpy.data.objects.new("PointCloud_Main", mesh)
            bpy.context.scene.collection.objects.link(pc_obj)
            
            # Add vertex colors
            if len(mesh_colors) == len(mesh.loops):
                color_layer = mesh.color_attributes.new(name="Col", domain='CORNER', type='FLOAT_COLOR')
                for i, c in enumerate(mesh_colors):
                    color_layer.data[i].color = c
            
            # ================== Point Cloud Material (with Emission) ==================
            mat = bpy.data.materials.new("PointCloudMat")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            
            for n in list(nodes):
                nodes.remove(n)
            
            out_node = nodes.new("ShaderNodeOutputMaterial")
            bsdf = nodes.new("ShaderNodeBsdfPrincipled")
            em_node = nodes.new("ShaderNodeEmission")
            attr_node = nodes.new("ShaderNodeAttribute")
            attr_node.attribute_name = "Col"
            
            links.new(attr_node.outputs["Color"], bsdf.inputs["Base Color"])
            links.new(attr_node.outputs["Color"], em_node.inputs["Color"])
            em_node.inputs["Strength"].default_value = 0.05
            
            add_node = nodes.new("ShaderNodeAddShader")
            links.new(bsdf.outputs["BSDF"], add_node.inputs[0])
            links.new(em_node.outputs["Emission"], add_node.inputs[1])
            links.new(add_node.outputs[0], out_node.inputs["Surface"])
            
            pc_obj.data.materials.append(mat)
            
            # ================== Load Gaussian Parameters and Create Ellipsoids ==================
            with open(gaussian_json, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            if "gaussian_params" in json_data:
                # Single-frame format with multiple objects
                print("[INFO] Detected single-frame gaussian_params format")
                gaussian_params = json_data["gaussian_params"]
                num_objects = json_data.get("num_objects", len(gaussian_params))
                
                print(f"[INFO] Found {num_objects} objects in single frame")
                
                cmap = matplotlib.colormaps['tab10']
                
                for obj_idx, (obj_id_str, obj_data) in enumerate(gaussian_params.items()):
                    obj_id = int(obj_id_str)
                    mean = np.array(obj_data["mean"], dtype=np.float32)
                    cov = np.array(obj_data["cov"], dtype=np.float32)
                    label = obj_data.get("label", "unknown")
                    
                    # Transform Gaussian parameters from OpenCV to Blender coordinate system
                    mean_blender = COORD_TRANSFORM @ mean
                    cov_blender = COORD_TRANSFORM @ cov @ COORD_TRANSFORM.T
                    
                    # Eigendecomposition
                    eigenvalues, eigenvectors = np.linalg.eigh(cov_blender)
                    eigenvalues = np.maximum(eigenvalues, 1e-8)
                    axes_lengths = self.ELLIPSOID_SCALE_FACTOR * np.sqrt(eigenvalues)
                    
                    # Create ellipsoid with high resolution
                    bpy.ops.mesh.primitive_uv_sphere_add(
                        segments=self.ELLIPSOID_SEGMENTS,
                        ring_count=self.ELLIPSOID_RINGS,
                        radius=1.0,
                        location=tuple(mean_blender)
                    )
                    ellipsoid_obj = bpy.context.active_object
                    ellipsoid_obj.name = f"Ellipsoid_{obj_id}"
                    
                    mesh_ellipsoid = ellipsoid_obj.data
                    mesh_ellipsoid.name = f"EllipsoidMesh_{obj_id}"
                    
                    # Apply transform
                    scale_matrix = np.diag(axes_lengths)
                    transform_matrix_3x3 = (eigenvectors @ scale_matrix).T
                    
                    transform_matrix_4x4 = np.eye(4, dtype=np.float32)
                    transform_matrix_4x4[:3, :3] = transform_matrix_3x3
                    
                    blender_matrix = mathutils.Matrix(transform_matrix_4x4.tolist())
                    loc, rot, scale = blender_matrix.decompose()
                    
                    ellipsoid_obj.rotation_quaternion = rot
                    ellipsoid_obj.scale = scale
                    
                    bpy.context.view_layer.update()
                    
                    # ================== Ellipsoid Material (with Fresnel) ==================
                    color_idx = obj_id - 1
                    color_rgb = cmap(color_idx % 10)[:3]
                    opacity = self.SNAPSHOT_OPACITY
                    
                    mat = bpy.data.materials.new(name=f"EllipsoidMat_{obj_id}")
                    mat.use_nodes = True
                    nodes = mat.node_tree.nodes
                    links = mat.node_tree.links
                    
                    for n in list(nodes):
                        nodes.remove(n)
                    
                    output_node = nodes.new("ShaderNodeOutputMaterial")
                    principled = nodes.new("ShaderNodeBsdfPrincipled")
                    
                    principled.inputs["Base Color"].default_value = (*color_rgb, 1.0)
                    principled.inputs["Alpha"].default_value = opacity
                    principled.inputs["Roughness"].default_value = 0.3
                    
                    # Add Fresnel effect for better edge visibility
                    fresnel_node = nodes.new("ShaderNodeFresnel")
                    fresnel_node.inputs["IOR"].default_value = 1.45
                    
                    colorramp = nodes.new("ShaderNodeValToRGB")
                    colorramp.color_ramp.elements[0].position = 0.0
                    colorramp.color_ramp.elements[0].color = (*color_rgb, 0.3 * opacity)
                    colorramp.color_ramp.elements[1].position = 1.0
                    colorramp.color_ramp.elements[1].color = (*color_rgb, 1.0 * opacity)
                    
                    links.new(fresnel_node.outputs["Fac"], colorramp.inputs["Fac"])
                    links.new(colorramp.outputs["Color"], principled.inputs["Alpha"])
                    links.new(principled.outputs["BSDF"], output_node.inputs["Surface"])
                    
                    mat.blend_method = 'BLEND'
                    ellipsoid_obj.data.materials.append(mat)
                    
                    print(f"[INFO] Created ellipsoid for object {obj_id} ({label})")
                    self.report({'INFO'}, f"Created ellipsoid for {label}")
                
                print(f"[INFO] Created {num_objects} ellipsoids")
            
            elif "frames" in json_data:
                # Multi-frame format - use first frame
                print("[INFO] Detected multi-frame format, using first frame")
                frames = json_data.get("frames", [])
                if frames:
                    first_frame = frames[0]
                    cmap = matplotlib.colormaps['tab10']
                    
                    for obj in first_frame.get("objects", []):
                        obj_id = obj["object_id"]
                        color_idx = obj.get("color_index", 0)
                        mean = np.array(obj["gaussian_3d"]["mean"], dtype=np.float32)
                        cov = np.array(obj["gaussian_3d"]["covariance"], dtype=np.float32)
                        
                        # Transform to Blender coordinates
                        mean_blender = COORD_TRANSFORM @ mean
                        cov_blender = COORD_TRANSFORM @ cov @ COORD_TRANSFORM.T
                        
                        eigenvalues, eigenvectors = np.linalg.eigh(cov_blender)
                        eigenvalues = np.maximum(eigenvalues, 1e-8)
                        axes_lengths = self.ELLIPSOID_SCALE_FACTOR * np.sqrt(eigenvalues)
                        
                        bpy.ops.mesh.primitive_uv_sphere_add(
                            segments=self.ELLIPSOID_SEGMENTS,
                            ring_count=self.ELLIPSOID_RINGS,
                            radius=1.0,
                            location=tuple(mean_blender)
                        )
                        ellipsoid_obj = bpy.context.active_object
                        ellipsoid_obj.name = f"Ellipsoid_{obj_id}"
                        
                        scale_matrix = np.diag(axes_lengths)
                        transform_matrix_3x3 = (eigenvectors @ scale_matrix).T
                        transform_matrix_4x4 = np.eye(4, dtype=np.float32)
                        transform_matrix_4x4[:3, :3] = transform_matrix_3x3
                        
                        blender_matrix = mathutils.Matrix(transform_matrix_4x4.tolist())
                        loc, rot, scale = blender_matrix.decompose()
                        ellipsoid_obj.rotation_quaternion = rot
                        ellipsoid_obj.scale = scale
                        
                        # Material with Fresnel
                        color_rgb = cmap(color_idx % 10)[:3]
                        opacity = self.SNAPSHOT_OPACITY
                        
                        mat = bpy.data.materials.new(name=f"EllipsoidMat_{obj_id}")
                        mat.use_nodes = True
                        nodes = mat.node_tree.nodes
                        links = mat.node_tree.links
                        
                        for n in list(nodes):
                            nodes.remove(n)
                        
                        output_node = nodes.new("ShaderNodeOutputMaterial")
                        principled = nodes.new("ShaderNodeBsdfPrincipled")
                        principled.inputs["Base Color"].default_value = (*color_rgb, 1.0)
                        principled.inputs["Roughness"].default_value = 0.3
                        
                        fresnel_node = nodes.new("ShaderNodeFresnel")
                        fresnel_node.inputs["IOR"].default_value = 1.45
                        
                        colorramp = nodes.new("ShaderNodeValToRGB")
                        colorramp.color_ramp.elements[0].position = 0.0
                        colorramp.color_ramp.elements[0].color = (*color_rgb, 0.3 * opacity)
                        colorramp.color_ramp.elements[1].position = 1.0
                        colorramp.color_ramp.elements[1].color = (*color_rgb, 1.0 * opacity)
                        
                        links.new(fresnel_node.outputs["Fac"], colorramp.inputs["Fac"])
                        links.new(colorramp.outputs["Color"], principled.inputs["Alpha"])
                        links.new(principled.outputs["BSDF"], output_node.inputs["Surface"])
                        
                        mat.blend_method = 'BLEND'
                        ellipsoid_obj.data.materials.append(mat)
                        
                        print(f"[INFO] Created ellipsoid for object {obj_id}")
            
            # ================== Setup Main Camera ==================
            self._setup_blender_camera(K, width_v, height_v)
            
            # Update status
            props.step1_complete = True
            props.step2_complete = True
            props.step3_complete = True
            props.status_message = "✓ Scene loaded! Now edit camera/object trajectories."
            
            # Set Blender timeline to match video_length
            bpy.context.scene.frame_start = 1
            bpy.context.scene.frame_end = props.video_length
            print(f"[INFO] Timeline set: frame 1 - {props.video_length}")
            
            print("[INFO] ===============================================")
            print("[INFO] Visualization completed!")
            print(f"[INFO] Total points: {len(background_verts)}")
            print("[INFO] ===============================================")
            
            self.report({'INFO'}, "Scene loaded successfully!")
            
            # Auto-switch to Material Preview shading for better visualization
            self._set_viewport_shading('MATERIAL')
            
        except Exception as e:
            import traceback
            self.report({'ERROR'}, f"Failed to load scene: {str(e)}")
            print(traceback.format_exc())
            return {'CANCELLED'}
        
        return {'FINISHED'}
    
    def _set_viewport_shading(self, shading_type='MATERIAL'):
        """Set viewport shading mode for all 3D views.
        
        Args:
            shading_type: One of 'WIREFRAME', 'SOLID', 'MATERIAL', 'RENDERED'
        """
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = shading_type
        print(f"[INFO] Viewport shading set to: {shading_type}")
    
    def _cleanup_old_objects(self):
        """Remove old objects, meshes, and materials."""
        # Remove objects
        prefixes = ["Camera_", "PointCloud", "CamPath", "Ellipsoid_", 
                    "EllipsoidPath_", "ObjectSnapshot_", "CameraFrame_", "RenderCamera"]
        for obj in list(bpy.data.objects):
            if any(obj.name.startswith(p) for p in prefixes) or obj.name in prefixes:
                bpy.data.objects.remove(obj, do_unlink=True)
        
        # Remove meshes
        mesh_prefixes = ["PointCloudMesh", "EllipsoidMesh_", "CameraFrustum_", 
                        "ObjectSnapshotMesh_", "CameraFrame_"]
        for mesh in list(bpy.data.meshes):
            if any(mesh.name.startswith(p) for p in mesh_prefixes):
                bpy.data.meshes.remove(mesh, do_unlink=True)
        
        # Remove materials
        mat_prefixes = ["PointCloudMat", "EllipsoidMat_", "CameraMat_", 
                       "ObjectSnapshotMat_", "CameraFrameMat_"]
        for mat in list(bpy.data.materials):
            if any(mat.name.startswith(p) for p in mat_prefixes):
                bpy.data.materials.remove(mat, do_unlink=True)
        
        # Remove curves
        for curve in list(bpy.data.curves):
            if curve.name == "CamPath":
                bpy.data.curves.remove(curve, do_unlink=True)
    
    def _setup_blender_camera(self, K, image_width, image_height):
        """Create and setup a Blender camera matching build_4d_control_scene.py."""
        cam_data = bpy.data.cameras.new(name='RenderCamera')
        
        fx, fy = K[0, 0], K[1, 1]
        
        cam_data.lens_unit = 'FOV'
        cam_data.angle = 2 * math.atan(image_width / (2 * fx))
        
        scene = bpy.context.scene
        scene.render.resolution_x = image_width
        scene.render.resolution_y = image_height
        
        cam_obj = bpy.data.objects.new('RenderCameraObj', cam_data)
        
        # Camera at origin, looking down Y axis (Blender convention)
        # Rotation to match OpenCV to Blender coordinate transform
        R_cam_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        
        cam_obj.location = mathutils.Vector((0, 0, 0))
        cam_obj.rotation_euler = mathutils.Matrix(R_cam_rotation).to_euler('XYZ')
        
        scene.collection.objects.link(cam_obj)
        scene.camera = cam_obj
        print("[INFO] Created and set up Blender camera.")


class VERSECRAFTER_OT_export_trajectories(Operator):
    """Export camera and object trajectories from Blender"""
    bl_idname = "versecrafter.export_trajectories"
    bl_label = "Export Trajectories"
    bl_description = "Export camera and 3D Gaussian trajectories to files"
    
    def execute(self, context):
        props = context.scene.versecrafter
        
        # Reset Step 3 progress when re-exporting trajectories
        # This ensures progress bar updates correctly when user iterates Step 2-3
        props.step5_complete = False
        props.step6_complete = False
        
        if not props.workflow_dir:
            self.report({'ERROR'}, "Please set a workflow directory")
            return {'CANCELLED'}
        
        try:
            if not HAS_DEPS:
                self.report({'ERROR'}, "numpy is required for export.")
                return {'CANCELLED'}
            
            scene = bpy.context.scene
            output_name = props.output_name or "camera_object_0"
            output_dir = os.path.join(props.workflow_dir, output_name)
            os.makedirs(output_dir, exist_ok=True)
            
            TOTAL_FRAMES = props.video_length
            
            # ================== Export Camera Trajectory ==================
            self.report({'INFO'}, "Exporting camera trajectory...")
            
            camera_objects = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
            
            if len(camera_objects) == 0:
                self.report({'ERROR'}, "No camera found in scene!")
                return {'CANCELLED'}
            
            cam_obj = camera_objects[0]
            extrinsics_list = []
            
            for i in range(TOTAL_FRAMES):
                frame_no = i + 1
                scene.frame_set(frame_no)
                
                # Get camera world matrix (camera-to-world in Blender)
                cam_matrix = cam_obj.matrix_world
                cam_matrix_np = np.array(cam_matrix, dtype=np.float32)
                extrinsics_list.append(cam_matrix_np)
            
            extrinsics_array = np.stack(extrinsics_list, axis=0)
            
            camera_npz_path = os.path.join(output_dir, "custom_camera_trajectory.npz")
            np.savez(camera_npz_path, extrinsics=extrinsics_array)
            self.report({'INFO'}, f"Camera trajectory saved: {camera_npz_path}")
            
            # ================== Export Ellipsoid Trajectories ==================
            self.report({'INFO'}, "Exporting ellipsoid trajectories...")
            
            ellipsoid_objects = [obj for obj in bpy.data.objects if obj.name.startswith("Ellipsoid_")]
            
            # Load color mapping from input JSON, or generate from object IDs
            input_json_path = os.path.join(props.workflow_dir, "fitted_3D_gaussian", "gaussian_params.json")
            obj_id_to_color_idx = {}
            
            if os.path.exists(input_json_path):
                with open(input_json_path, 'r') as f:
                    input_data = json.load(f)
                if "obj_id_to_color_idx" in input_data:
                    obj_id_to_color_idx = {str(k): v for k, v in input_data["obj_id_to_color_idx"].items()}
                elif "gaussian_params" in input_data:
                    # Generate color mapping from gaussian_params keys (object IDs)
                    # This matches the logic in export_blender_custom_trajectories.py
                    for obj_id_str in input_data["gaussian_params"].keys():
                        try:
                            obj_id = int(obj_id_str)
                            # Use (obj_id - 1) as color index to match tab10/tab20 colormap
                            obj_id_to_color_idx[obj_id_str] = obj_id - 1
                        except ValueError:
                            obj_id_to_color_idx[obj_id_str] = 0
                    print(f"[INFO] Generated color mapping for {len(obj_id_to_color_idx)} objects")
            
            # Prepare export data
            export_data = {
                "metadata": {
                    "num_objects": len(ellipsoid_objects),
                    "num_frames": TOTAL_FRAMES,
                    "frame_step": 1,
                    "description": "Exported from Blender VerseCrafter addon",
                    "obj_id_to_color_idx": obj_id_to_color_idx
                },
                "frames": []
            }
            
            for i in range(TOTAL_FRAMES):
                frame_no = i + 1
                scene.frame_set(frame_no)
                
                frame_data = {
                    "frame_index": i,
                    "objects": []
                }
                
                for ellipsoid_obj in ellipsoid_objects:
                    obj_id_str = ellipsoid_obj.name.replace("Ellipsoid_", "")
                    
                    mesh = ellipsoid_obj.data
                    if len(mesh.vertices) > 0:
                        world_matrix = ellipsoid_obj.matrix_world
                        
                        # Get center
                        center = np.array([
                            world_matrix[0][3],
                            world_matrix[1][3],
                            world_matrix[2][3]
                        ], dtype=np.float32)
                        
                        # Compute covariance from transform
                        transform_3x3 = np.array([
                            [world_matrix[0][0], world_matrix[0][1], world_matrix[0][2]],
                            [world_matrix[1][0], world_matrix[1][1], world_matrix[1][2]],
                            [world_matrix[2][0], world_matrix[2][1], world_matrix[2][2]]
                        ], dtype=np.float32)
                        
                        U, S, Vt = np.linalg.svd(transform_3x3)
                        eigenvalues = (S / 2.0) ** 2
                        covariance = U @ np.diag(eigenvalues) @ U.T
                        
                        # Get color index
                        color_idx = obj_id_to_color_idx.get(obj_id_str, int(obj_id_str) if obj_id_str.isdigit() else 0)
                        
                        object_data = {
                            "object_id": obj_id_str,
                            "color_index": color_idx,
                            "gaussian_3d": {
                                "mean": center.tolist(),
                                "covariance": covariance.tolist()
                            }
                        }
                        frame_data["objects"].append(object_data)
                
                export_data["frames"].append(frame_data)
            
            gaussian_json_path = os.path.join(output_dir, "custom_3D_gaussian_trajectory.json")
            with open(gaussian_json_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.report({'INFO'}, f"Ellipsoid trajectories saved: {gaussian_json_path}")
            
            # ================== Upload to Server ==================
            server_url = props.server_url.rstrip('/')
            # Get server preprocess dir from class variable
            server_preprocess_dir = VERSECRAFTER_OT_preprocess._server_output_dir
            
            if server_preprocess_dir and server_url:
                self.report({'INFO'}, "Uploading trajectories to server...")
                props.status_message = "Uploading trajectories to server..."
                
                # Define server trajectory directory
                server_trajectory_dir = f"{server_preprocess_dir}/trajectories_{output_name}"
                
                # Upload camera trajectory
                success, result = upload_file_to_server(server_url, camera_npz_path, server_trajectory_dir)
                if not success:
                    self.report({'WARNING'}, f"Failed to upload camera trajectory: {result}")
                    props.status_message = f"⚠ Local export done, but upload failed: {result}"
                else:
                    self.report({'INFO'}, f"Camera trajectory uploaded to: {result}")
                    
                    # Upload gaussian trajectory
                    success, result = upload_file_to_server(server_url, gaussian_json_path, server_trajectory_dir)
                    if not success:
                        self.report({'WARNING'}, f"Failed to upload gaussian trajectory: {result}")
                        props.status_message = f"⚠ Partial upload, gaussian failed: {result}"
                    else:
                        self.report({'INFO'}, f"Gaussian trajectory uploaded to: {result}")
                        
                        # Store trajectory dir for postprocess
                        VERSECRAFTER_OT_postprocess._server_trajectory_dir = server_trajectory_dir
                        
                        props.step4_complete = True
                        props.status_message = f"✓ Trajectories exported and uploaded!"
                        self.report({'INFO'}, "Export and upload complete!")
            else:
                # No server configured or preprocess not run, just local export
                props.step4_complete = True
                props.status_message = f"✓ Trajectories exported to {output_dir}"
                self.report({'INFO'}, f"Export complete! Files saved to: {output_dir}")
                if not server_preprocess_dir:
                    self.report({'WARNING'}, "Server preprocess dir not found. Run preprocessing first for auto-upload.")
            
        except Exception as e:
            import traceback
            self.report({'ERROR'}, f"Failed to export trajectories: {str(e)}")
            print(traceback.format_exc())
            return {'CANCELLED'}
        
        return {'FINISHED'}


def start_render_on_server(server_url: str, preprocess_dir: str, trajectory_dir: str, video_length: int) -> tuple:
    """Start rendering control maps on server via GET request."""
    try:
        params = urllib.parse.urlencode({
            'preprocess_dir': preprocess_dir,
            'trajectory_dir': trajectory_dir,
            'video_length': video_length
        })
        url = f"{server_url}/api/render_get?{params}"
        
        with _make_request(url, timeout=30) as response:
            response_text = response.read().decode('utf-8')
            
            # Check for HTML response
            if response_text.strip().startswith('<'):
                return False, "Server returned HTML (proxy blocking). Login in browser first."
            
            result = json.loads(response_text)
            if 'id' in result:
                return True, result['id']
            else:
                return False, result.get('error', 'No task_id returned')
    except json.JSONDecodeError:
        return False, "Server returned invalid JSON (proxy might be blocking)"
    except Exception as e:
        return False, str(e)


def start_generate_on_server(server_url: str, preprocess_dir: str, control_map_dir: str, 
                             video_prompt: str, video_length: int, output_dir: str,
                             num_inference_steps: int = 50, guidance_scale: float = 5.0,
                             seed: int = 2025, fps: int = 16) -> tuple:
    """Start video generation on server via GET request."""
    try:
        params = urllib.parse.urlencode({
            'preprocess_dir': preprocess_dir,
            'control_map_dir': control_map_dir,
            'video_prompt': video_prompt,
            'video_length': video_length,
            'output_dir': output_dir,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'seed': seed,
            'fps': fps
        })
        url = f"{server_url}/api/generate_get?{params}"
        
        with _make_request(url, timeout=30) as response:
            response_text = response.read().decode('utf-8')
            
            # Check for HTML response
            if response_text.strip().startswith('<'):
                return False, "Server returned HTML (proxy blocking). Login in browser first."
            
            result = json.loads(response_text)
            if 'id' in result:
                return True, result['id']
            else:
                return False, result.get('error', 'No task_id returned')
    except json.JSONDecodeError:
        return False, "Server returned invalid JSON (proxy might be blocking)"
    except Exception as e:
        return False, str(e)


class VERSECRAFTER_OT_postprocess(Operator):
    """Run VerseCrafter video generation (export + render + generate) on GPU server"""
    bl_idname = "versecrafter.postprocess"
    bl_label = "Generate Video"
    bl_description = "Export trajectories, render control maps, generate video, download result"
    
    _timer = None
    _task_id = None
    _stage = None  # "upload", "render", "generate"
    _server_trajectory_dir = None
    _server_preprocess_dir = None
    _server_control_map_dir = None
    _server_output_dir = None
    _poll_count = 0  # Track polling count for adaptive interval
    
    def execute(self, context):
        props = context.scene.versecrafter
        
        # Reset Step 3 progress for new video generation iteration
        # Step 2 (step4_complete) is preserved since trajectories were already exported
        props.step5_complete = False
        props.step6_complete = False
        
        # Check if trajectories were exported in Step 2
        if not props.step4_complete:
            self.report({'ERROR'}, "Please export trajectories in Step 2 first")
            return {'CANCELLED'}
        
        # Find trajectory files (already exported in Step 2)
        output_name = props.output_name or "camera_object_0"
        output_dir = os.path.join(props.workflow_dir, output_name)
        
        camera_traj = os.path.join(output_dir, "custom_camera_trajectory.npz")
        gaussian_traj = os.path.join(output_dir, "custom_3D_gaussian_trajectory.json")
        
        if not os.path.exists(camera_traj):
            self.report({'ERROR'}, f"Camera trajectory not found: {camera_traj}. Please export trajectories in Step 2.")
            return {'CANCELLED'}
        
        if not os.path.exists(gaussian_traj):
            self.report({'ERROR'}, f"Gaussian trajectory not found: {gaussian_traj}. Please export trajectories in Step 2.")
            return {'CANCELLED'}
        
        server_url = props.server_url.rstrip('/')
        
        # Define server-side paths
        image_filename = os.path.basename(props.input_image_path) if props.input_image_path else "image"
        base_name = os.path.splitext(image_filename)[0]
        timestamp = int(time.time())
        
        # Use the same preprocess output dir from the preprocess step
        server_preprocess_dir = VERSECRAFTER_OT_preprocess._server_output_dir
        if not server_preprocess_dir:
            self.report({'ERROR'}, "Preprocess output directory not found. Please run Step 1 preprocessing first.")
            return {'CANCELLED'}
        
        server_trajectory_dir = f"{server_preprocess_dir}/trajectories_{output_name}"
        # Control map dir is inside the trajectory folder
        server_control_map_dir = f"{server_trajectory_dir}/rendered_4D_control_maps"
        # Save generated video in the same folder as trajectory files
        server_output_dir = server_trajectory_dir
        
        VERSECRAFTER_OT_postprocess._server_preprocess_dir = server_preprocess_dir
        VERSECRAFTER_OT_postprocess._server_trajectory_dir = server_trajectory_dir
        VERSECRAFTER_OT_postprocess._server_control_map_dir = server_control_map_dir
        VERSECRAFTER_OT_postprocess._server_output_dir = server_output_dir
        
        props.is_processing = True
        props.status_message = "Uploading trajectory files..."
        
        # Upload trajectory files
        self.report({'INFO'}, "Uploading camera trajectory...")
        success, result = upload_file_to_server(server_url, camera_traj, server_trajectory_dir)
        if not success:
            props.is_processing = False
            self.report({'ERROR'}, f"Failed to upload camera trajectory: {result}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Camera trajectory uploaded to: {result}")
        
        self.report({'INFO'}, "Uploading gaussian trajectory...")
        success, result = upload_file_to_server(server_url, gaussian_traj, server_trajectory_dir)
        if not success:
            props.is_processing = False
            self.report({'ERROR'}, f"Failed to upload gaussian trajectory: {result}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Gaussian trajectory uploaded to: {result}")
        
        # Start rendering on server
        props.status_message = "Starting control map rendering on server..."
        VERSECRAFTER_OT_postprocess._stage = "render"
        
        success, task_id = start_render_on_server(
            server_url,
            server_preprocess_dir,
            server_trajectory_dir,
            props.video_length
        )
        
        if not success:
            props.is_processing = False
            self.report({'ERROR'}, f"Failed to start rendering: {task_id}")
            return {'CANCELLED'}
        
        VERSECRAFTER_OT_postprocess._task_id = task_id
        self.report({'INFO'}, f"Rendering started, task ID: {task_id}")
        props.status_message = f"Rendering control maps... (Task: {task_id[:8]}...)"
        
        # Register timer to check completion
        VERSECRAFTER_OT_postprocess._poll_count = 0
        VERSECRAFTER_OT_postprocess._timer = bpy.app.timers.register(
            self._check_postprocess_completion,
            first_interval=3.0
        )
        
        return {'FINISHED'}
    
    @staticmethod
    def _check_postprocess_completion():
        """Timer callback to check postprocessing status."""
        props = bpy.context.scene.versecrafter
        server_url = props.server_url.rstrip('/')
        task_id = VERSECRAFTER_OT_postprocess._task_id
        stage = VERSECRAFTER_OT_postprocess._stage
        
        if not task_id:
            props.is_processing = False
            return None
        
        # Increment poll count for adaptive interval
        VERSECRAFTER_OT_postprocess._poll_count += 1
        poll_count = VERSECRAFTER_OT_postprocess._poll_count
        
        # Adaptive polling: start at 5s, increase to max 15s for long-running video generation
        # Render stage: 5-10s interval
        # Generate stage: 5-15s interval (video generation takes longer)
        if stage == "generate":
            interval = min(5.0 + poll_count * 1.0, 15.0)
        else:
            interval = min(5.0 + poll_count * 0.5, 10.0)
        
        # Check task status
        status = check_task_status(server_url, task_id)
        
        if status.get('status') == 'running':
            progress = status.get('progress', 0) * 100
            message = status.get('message', 'Processing...')
            stage_name = "Rendering" if stage == "render" else "Generating video"
            props.status_message = f"{stage_name}: {message} ({progress:.0f}%)"
            return interval
        
        elif status.get('status') == 'completed':
            if stage == "render":
                # Rendering complete, start video generation
                props.status_message = "Control maps rendered! Starting video generation..."
                VERSECRAFTER_OT_postprocess._stage = "generate"
                VERSECRAFTER_OT_postprocess._poll_count = 0  # Reset poll count for new stage
                
                success, task_id = start_generate_on_server(
                    server_url,
                    VERSECRAFTER_OT_postprocess._server_preprocess_dir,
                    VERSECRAFTER_OT_postprocess._server_control_map_dir,
                    props.video_prompt,
                    props.video_length,
                    VERSECRAFTER_OT_postprocess._server_output_dir,
                    props.num_inference_steps,
                    props.guidance_scale,
                    props.seed,
                    props.fps
                )
                
                if success:
                    VERSECRAFTER_OT_postprocess._task_id = task_id
                    props.status_message = f"Generating video... (Task: {task_id[:8]}...)"
                    return 5.0  # First poll for generate stage
                else:
                    props.status_message = f"✗ Failed to start generation: {task_id}"
                    props.is_processing = False
                    return None
            
            else:
                # Video generation complete, download the video
                props.status_message = "Video generated! Downloading..."
                
                result_data = status.get('result', {})
                video_path = result_data.get('video_path') if result_data else None
                
                if not video_path:
                    # Fallback: try common video file names
                    server_output_dir = VERSECRAFTER_OT_postprocess._server_output_dir
                    for filename in ['generated_video_0.mp4', 'generated_video.mp4']:
                        video_path = f"{server_output_dir}/{filename}"
                        break
                
                # Download video to the same folder as trajectory files
                output_name = props.output_name or "camera_object_0"
                local_video_dir = os.path.join(props.workflow_dir, output_name)
                os.makedirs(local_video_dir, exist_ok=True)
                # Use the same filename as the server
                video_filename = os.path.basename(video_path) if video_path else "generated_video.mp4"
                local_video_path = os.path.join(local_video_dir, video_filename)
                
                success, result = download_file_from_server(server_url, video_path, local_video_path)
                
                if success:
                    props.step5_complete = True
                    props.step6_complete = True
                    props.status_message = f"✓ Video saved to: {local_video_path}"
                else:
                    props.status_message = f"⚠ Video generated but download failed: {result}"
                
                props.is_processing = False
                VERSECRAFTER_OT_postprocess._task_id = None
                VERSECRAFTER_OT_postprocess._stage = None
                return None
        
        elif status.get('status') == 'failed':
            props.status_message = f"✗ Failed: {status.get('error', 'Unknown error')}"
            props.is_processing = False
            VERSECRAFTER_OT_postprocess._task_id = None
            return None
        
        else:
            # Unknown status or pending, keep checking with adaptive interval
            return interval


class VERSECRAFTER_OT_test_connection(Operator):
    """Test connection to the server"""
    bl_idname = "versecrafter.test_connection"
    bl_label = "Test Connection"
    bl_description = "Test connection to the VerseCrafter API server"
    
    def execute(self, context):
        props = context.scene.versecrafter
        
        if not props.server_url:
            self.report({'ERROR'}, "Please set the server URL")
            return {'CANCELLED'}
        
        server_url = props.server_url.rstrip('/')
        
        # Set global cookie if provided
        if props.proxy_cookie:
            set_proxy_cookie(props.proxy_cookie)
        
        self.report({'INFO'}, f"Testing connection to {server_url}...")
        
        success, message = test_server_connection(server_url, props.proxy_cookie)
        
        if success:
            props.status_message = f"✓ {message}"
            self.report({'INFO'}, message)
        else:
            props.status_message = f"✗ {message}"
            self.report({'ERROR'}, message)
        
        return {'FINISHED'}


class VERSECRAFTER_OT_dummy_progress(Operator):
    """Dummy operator for progress bar display (non-interactive)"""
    bl_idname = "versecrafter.dummy_progress"
    bl_label = ""
    bl_description = "Progress indicator"
    bl_options = {'INTERNAL'}
    
    def execute(self, context):
        # This operator does nothing - it's just for visual display
        return {'FINISHED'}


class VERSECRAFTER_OT_open_folder(Operator):
    """Open workflow folder in file browser"""
    bl_idname = "versecrafter.open_folder"
    bl_label = "Open Folder"
    bl_description = "Open the workflow folder in the system file browser"
    
    folder_type: StringProperty(default="workflow")
    
    def execute(self, context):
        props = context.scene.versecrafter
        
        if self.folder_type == "workflow":
            folder = props.workflow_dir
        elif self.folder_type == "output":
            output_name = props.output_name or "camera_object_0"
            folder = os.path.join(props.workflow_dir, output_name)
        elif self.folder_type == "video":
            # Open the trajectory directory where videos are now saved
            output_name = props.output_name or "camera_object_0"
            folder = os.path.join(props.workflow_dir, output_name)
        else:
            folder = props.workflow_dir
        
        if not folder or not os.path.exists(folder):
            self.report({'ERROR'}, f"Folder not found: {folder}")
            return {'CANCELLED'}
        
        # Open folder in system file browser
        import platform
        system = platform.system()
        
        try:
            if system == "Windows":
                os.startfile(folder)
            elif system == "Darwin":  # macOS
                subprocess.Popen(["open", folder])
            else:  # Linux
                subprocess.Popen(["xdg-open", folder])
            
            self.report({'INFO'}, f"Opened: {folder}")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to open folder: {str(e)}")
            return {'CANCELLED'}
        
        return {'FINISHED'}

