import io, time, json, os, queue, sys, urllib.request
from flask import Flask, Response, render_template, request, jsonify

from skyllm.robot_info import RobotInfo
from skyllm.utils import print_t, CURRENT_PROJ_DIR
from skyllm.llm_controller import LLMController, _USER_LOG_QUEUE

class SkyLLM:
    def __init__(self, robot_info: RobotInfo):
        self.llm_controller = LLMController(robot_info)
        self.running = True
        self.app = Flask(__name__, 
                        template_folder=os.path.join(CURRENT_PROJ_DIR, 'assets'))
        self.setup_routes()

    def setup_routes(self):
        """Sets up the Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/chat', methods=['POST'])
        def chat():
            """Handle chat messages and stream responses using SSE."""
            data = request.get_json()
            user_message = data.get('message', '')
            
            if not user_message:
                return jsonify({'type': 'text', 'content': '[WARNING] Empty command!'})
            
            if user_message == "exit":
                self.running = False
                return jsonify({'type': 'text', 'content': 'Shutting down...'})
            
            # Send instruction to LLM Controller
            self.llm_controller.put_instruction(user_message)
            
            def generate():
                # Send initial acknowledgment
                yield f"data: {json.dumps({'type': 'text', 'content': 'Okay! Working on it...'})}\n\n"
                
                # Stream messages from the queue as they arrive
                while True:
                    try:
                        msg = _USER_LOG_QUEUE.get(timeout=3.0)
                        if msg == '#end':
                            print_t("[UI] End of plan")
                            return "data: [DONE]\n\n"
                    
                    except queue.Empty:
                        continue

                    print_t(f"[UI] New message: {msg}")
                    msg_str = str(msg)
                    
                    # Check if message contains an image (base64 encoded)
                    if '<img src="data:image/' in msg_str:
                        # For images, we need to ensure JSON encoding doesn't break the HTML
                        response_data = json.dumps({'type': 'image', 'content': msg_str}, ensure_ascii=False)
                        yield f"data: {response_data}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'text', 'content': msg_str})}\n\n"
            
            return Response(generate(), mimetype='text/event-stream')
        
        @self.app.route('/robot-pov/')
        def video_feed_pov():
            """Stream robot POV video feed."""
            return Response(
                self.generate_mjpeg_stream('pov'),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        @self.app.route('/health')
        def health():
            """Health check; also reports whether the vision service is reachable."""
            from skyllm.yolo_client import EDGE_SERVICE_IP, EDGE_SERVICE_PORT
            vision_ok = False
            try:
                with urllib.request.urlopen(
                    f"http://{EDGE_SERVICE_IP}:{EDGE_SERVICE_PORT}/health", timeout=1.0
                ) as r:
                    vision_ok = (r.status == 200)
            except Exception:
                vision_ok = False
            robot_label = getattr(self.llm_controller.robot.robot_info, "robot_type", "uav")
            return jsonify({'status': 'running', 'robot': robot_label, 'vision_service': vision_ok})

        @self.app.route('/telemetry')
        def telemetry():
            """Returns current live telemetry from the robot."""
            fn = getattr(self.llm_controller.robot, "get_telemetry_data", None)
            if fn:
                return jsonify(fn())
            return jsonify({})

        @self.app.route('/motors/set', methods=['POST'])
        def set_motors():
            """Sets motor PWM values from UI sliders directly."""
            data = request.get_json() or {}
            m1 = data.get('m1', 1000)
            m2 = data.get('m2', 1000)
            m3 = data.get('m3', 1000)
            m4 = data.get('m4', 1000)
            fn = getattr(self.llm_controller.robot, "set_manual_motors", None)
            if fn:
                res = fn(m1, m2, m3, m4)
                return jsonify({'status': 'ok', 'motors': res})
            return jsonify({'status': 'error', 'message': 'Not supported on this platform'}), 400

        @self.app.route('/connection/status', methods=['GET'])
        def connection_status():
            """Returns available connection profiles and active status."""
            from skyllm.connection_manager import get_connection_status
            robot = getattr(self.llm_controller, "robot", None)
            return jsonify(get_connection_status(robot))

        @self.app.route('/connection/switch', methods=['POST'])
        def connection_switch():
            """Switches connection mode (usb, radio, sim), saves to config, and reconnects."""
            from skyllm.connection_manager import switch_connection_mode
            data = request.get_json() or {}
            mode = data.get('mode') or data.get('address') or 'radio'
            robot = getattr(self.llm_controller, "robot", None)
            res = switch_connection_mode(robot, mode)
            return jsonify(res)

        @self.app.route('/connection/reconnect', methods=['POST'])
        def connection_reconnect():
            """Forces reconnection on the active connection."""
            from skyllm.connection_manager import switch_connection_mode, get_current_configured_address
            robot = getattr(self.llm_controller, "robot", None)
            addr = getattr(robot, "system_address", get_current_configured_address())
            res = switch_connection_mode(robot, addr)
            return jsonify(res)

    def generate_mjpeg_stream(self, source: str):
        """Generate MJPEG stream for video feeds."""
        while self.running:
            if source == 'pov':
                frame = self.llm_controller.fetch_robot_pov()
            else:
                frame = None
            
            if frame is None:
                time.sleep(1.0 / 30.0)
                continue
                
            buf = io.BytesIO()
            frame.save(buf, format='JPEG')
            buf.seek(0)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.read() + b'\r\n')
            time.sleep(1.0 / 30.0)

    def run(self):
        """Start the SkyLLM system with Flask server."""
        # Start the LLM controller
        self.llm_controller.start_controller()

        # Start the Flask server
        print_t("[SkyLLM] Starting Ground Control Station on http://127.0.0.1:50000")
        self.app.run(host='127.0.0.1', port=50000, debug=False, threaded=True)
        
        # When Flask stops, stop the LLM controller
        self.llm_controller.stop_controller()

def main():
    if not os.environ.get("GROQ_API_KEY"):
        print_t("[SkyLLM] GROQ_API_KEY is not set. Copy .env.example to .env and add your "
                "Groq key (cp .env.example .env), or run: export GROQ_API_KEY=gsk_...")
        sys.exit(1)
    with open(os.path.join(CURRENT_PROJ_DIR, 'config/robot_info.json'), 'r') as f:
        app = SkyLLM(RobotInfo.from_dict(json.load(f)))
        app.run()

if __name__ == '__main__':
    main()