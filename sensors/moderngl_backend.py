"""ModernGL render backend: static lot + dynamic ego + OpenCV fisheye / pinhole."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

import glm
import moderngl
import numpy as np
from PIL import Image

from scene.ego_vehicle import build_ego_mesh
from scene.lot import build_static_lot
from sensors.fisheye_opencv import remap_pinhole_to_fisheye

ROOT = Path(__file__).resolve().parents[1]


SCENE_VS = """
#version 330 core
in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;
uniform mat4 u_mvp;
uniform mat4 u_model;
out vec3 v_normal;
out vec3 v_color;
out vec3 v_world;
void main() {
    vec4 world = u_model * vec4(in_position, 1.0);
    v_world = world.xyz;
    v_normal = mat3(u_model) * in_normal;
    v_color = in_color;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

SCENE_FS = """
#version 330 core
in vec3 v_normal;
in vec3 v_color;
in vec3 v_world;
uniform vec3 u_light_dir;
uniform vec3 u_cam_pos;
out vec4 f_color;
void main() {
    vec3 n = normalize(v_normal);
    float ndl = max(dot(n, normalize(-u_light_dir)), 0.0);
    float ambient = 0.42;
    float diffuse = 0.78 * ndl;
    float dist = length(v_world - u_cam_pos);
    float fog = clamp((dist - 8.0) / 40.0, 0.0, 0.55);
    vec3 lit = v_color * (ambient + diffuse);
    vec3 fog_col = vec3(0.28, 0.30, 0.33);
    f_color = vec4(mix(lit, fog_col, fog), 1.0);
}
"""

# Legacy Phase 0/1 equidistant remap (kept for fisheye_mode=equidistant)
FISHEYE_VS = """
#version 330 core
in vec2 in_vert;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

FISHEYE_FS = """
#version 330 core
in vec2 v_uv;
uniform sampler2D u_tex;
uniform float u_pinhole_fov_y;
uniform float u_fisheye_fov;
uniform float u_aspect;
out vec4 f_color;
void main() {
    vec2 uv = v_uv * 2.0 - 1.0;
    uv.x *= u_aspect;
    float r = length(uv);
    if (r > 1.0) {
        f_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    float half_ffov = u_fisheye_fov * 0.5;
    float theta = r * half_ffov;
    float phi = atan(uv.y, uv.x);
    vec3 dir = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), -cos(theta));
    float half_yfov = u_pinhole_fov_y * 0.5;
    float tan_half_y = tan(half_yfov);
    float tan_half_x = tan_half_y * u_aspect;
    if (dir.z >= -1e-4) {
        f_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    vec2 pin;
    pin.x = (dir.x / -dir.z) / tan_half_x;
    pin.y = (dir.y / -dir.z) / tan_half_y;
    vec2 sample_uv = pin * 0.5 + 0.5;
    if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
        f_color = vec4(0.02, 0.02, 0.03, 1.0);
        return;
    }
    f_color = texture(u_tex, sample_uv);
}
"""


def _mat4_bytes(m: glm.mat4) -> bytes:
    return np.array(m, dtype="f4").T.tobytes()


def flu_ego_model(x: float, y: float, z: float, yaw_rad: float) -> glm.mat4:
    """Model matrix: local GL ego mesh → world GL from world-FLU pose."""
    t = glm.vec3(-y, z, x)
    return glm.translate(t) * glm.rotate(-float(yaw_rad), glm.vec3(0.0, 1.0, 0.0))


class ModernGLFisheyeBackend:
    """Offscreen renderer with static lot VAO + dynamic ego VAO.

    Default fisheye path (Phase 2): wide pinhole FBO → OpenCV Kannala–Brandt
    LUT (config K/D) → cv2.remap. Legacy equidistant GPU remap still available
    via fisheye_mode='equidistant'.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 960,
        pinhole_fov_y_deg: float = 165.0,
        fisheye_fov_deg: float = 190.0,
        vehicle_cfg: Optional[dict] = None,
        assets_root: Optional[Path] = None,
        fisheye_mode: str = "opencv",
        near: float = 0.05,
        far: float = 80.0,
    ):
        self.width = int(width)
        self.height = int(height)
        self.pinhole_fov_y_deg = float(pinhole_fov_y_deg)
        self.fisheye_fov = math.radians(float(fisheye_fov_deg))
        self.aspect = self.width / self.height
        self.fisheye_mode = str(fisheye_mode or "opencv").lower()
        self.assets_root = Path(assets_root) if assets_root else ROOT / "assets"
        self.near = float(near)
        self.far = float(far)

        vcfg = vehicle_cfg or {}
        length_m = float(vcfg.get("length_m", 4.4))
        width_m = float(vcfg.get("width_m", 1.9))
        height_m = float(vcfg.get("height_m", 1.5))

        self.ctx = moderngl.create_context(standalone=True)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        self._prog = self.ctx.program(vertex_shader=SCENE_VS, fragment_shader=SCENE_FS)

        lot_data = build_static_lot(self.assets_root)
        ego_data = build_ego_mesh(length_m, width_m, height_m, self.assets_root)

        self._lot_vbo = self.ctx.buffer(lot_data.tobytes())
        self._lot_vao = self.ctx.vertex_array(
            self._prog,
            [(self._lot_vbo, "3f 3f 3f", "in_position", "in_normal", "in_color")],
        )
        self._ego_vbo = self.ctx.buffer(ego_data.tobytes())
        self._ego_vao = self.ctx.vertex_array(
            self._prog,
            [(self._ego_vbo, "3f 3f 3f", "in_position", "in_normal", "in_color")],
        )

        self._color_tex = self.ctx.texture((self.width, self.height), 3)
        depth_rb = self.ctx.depth_renderbuffer((self.width, self.height))
        self._fbo = self.ctx.framebuffer(
            color_attachments=[self._color_tex], depth_attachment=depth_rb
        )

        self._fish_tex = self.ctx.texture((self.width, self.height), 3)
        self._fish_fbo = self.ctx.framebuffer(color_attachments=[self._fish_tex])
        self._fish_prog = self.ctx.program(
            vertex_shader=FISHEYE_VS, fragment_shader=FISHEYE_FS
        )
        quad = np.array(
            [-1, -1, 0, 0, 1, -1, 1, 0, -1, 1, 0, 1, 1, 1, 1, 1], dtype="f4"
        )
        quad_vbo = self.ctx.buffer(quad.tobytes())
        self._fish_vao = self.ctx.vertex_array(
            self._fish_prog, [(quad_vbo, "2f 2f", "in_vert", "in_uv")]
        )
        self._color_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._color_tex.repeat_x = False
        self._color_tex.repeat_y = False

        self._lot_model = glm.mat4(1.0)
        self._ego_model = glm.mat4(1.0)
        self._light_dir = (0.35, -1.0, 0.25)
        self._proj_fisheye_pinhole = glm.perspective(
            glm.radians(self.pinhole_fov_y_deg), self.aspect, self.near, self.far
        )
        self._released = False

    @property
    def gl_info(self) -> str:
        return "%s | %s" % (
            self.ctx.info.get("GL_RENDERER"),
            self.ctx.info.get("GL_VERSION"),
        )

    def set_ego_pose(self, x: float, y: float, z: float, yaw_rad: float) -> None:
        """Update dynamic ego model matrix from world-FLU pose."""
        self._ego_model = flu_ego_model(x, y, z, yaw_rad)

    def _draw_scene(self, proj: glm.mat4, view: glm.mat4, eye: Sequence[float]) -> None:
        self._fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.clear(0.28, 0.30, 0.33, 1.0)
        self._prog["u_light_dir"].value = self._light_dir
        self._prog["u_cam_pos"].value = (float(eye[0]), float(eye[1]), float(eye[2]))

        for model, vao in (
            (self._lot_model, self._lot_vao),
            (self._ego_model, self._ego_vao),
        ):
            mvp = proj * view * model
            self._prog["u_mvp"].write(_mat4_bytes(mvp))
            self._prog["u_model"].write(_mat4_bytes(model))
            vao.render(moderngl.TRIANGLES)

    def _read_fbo_rgb(self, fbo) -> np.ndarray:
        data = fbo.read(components=3, alignment=1)
        img = Image.frombytes("RGB", (self.width, self.height), data).transpose(
            Image.FLIP_TOP_BOTTOM
        )
        return np.asarray(img, dtype=np.uint8)

    def _render_wide_pinhole(
        self,
        eye: Sequence[float],
        target: Sequence[float],
        up: Sequence[float],
    ) -> np.ndarray:
        view = glm.lookAt(glm.vec3(*eye), glm.vec3(*target), glm.vec3(*up))
        self._draw_scene(self._proj_fisheye_pinhole, view, eye)
        return self._read_fbo_rgb(self._fbo)

    def _remap_equidistant_gpu(self) -> np.ndarray:
        self._fish_fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self._color_tex.use(location=0)
        self._fish_prog["u_tex"].value = 0
        self._fish_prog["u_pinhole_fov_y"].value = math.radians(self.pinhole_fov_y_deg)
        self._fish_prog["u_fisheye_fov"].value = self.fisheye_fov
        self._fish_prog["u_aspect"].value = self.aspect
        self.ctx.disable(moderngl.DEPTH_TEST)
        self._fish_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.DEPTH_TEST)
        return self._read_fbo_rgb(self._fish_fbo)

    def render_camera(
        self,
        eye: Sequence[float],
        target: Sequence[float],
        up: Sequence[float],
        K: Optional[np.ndarray] = None,
        D: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Render one fisheye view; returns HxWx3 uint8 RGB (top-left origin).

        When fisheye_mode is 'opencv' (default) and K,D are provided, uses the
        OpenCV Kannala–Brandt model from config as ground-truth intrinsics.
        """
        if self._released:
            raise RuntimeError("ModernGLFisheyeBackend already released")

        use_opencv = self.fisheye_mode in ("opencv", "cv2", "kannala", "kb")
        if use_opencv and K is not None and D is not None:
            pinhole = self._render_wide_pinhole(eye, target, up)
            return remap_pinhole_to_fisheye(
                pinhole,
                K=K,
                D=D,
                pinhole_fov_y_deg=self.pinhole_fov_y_deg,
                out_size=(self.width, self.height),
            )

        # Legacy equidistant GPU path (Phase 0/1)
        self._render_wide_pinhole(eye, target, up)
        return self._remap_equidistant_gpu()

    def render_pinhole(
        self,
        eye: Sequence[float],
        target: Sequence[float],
        up: Sequence[float],
        fov_y_deg: float = 55.0,
        near: float = 0.1,
        far: float = 120.0,
    ) -> np.ndarray:
        """Render a pinhole (no fisheye) view — used for god/orbit overview."""
        if self._released:
            raise RuntimeError("ModernGLFisheyeBackend already released")
        proj = glm.perspective(glm.radians(float(fov_y_deg)), self.aspect, near, far)
        view = glm.lookAt(glm.vec3(*eye), glm.vec3(*target), glm.vec3(*up))
        self._draw_scene(proj, view, eye)
        return self._read_fbo_rgb(self._fbo)

    def release(self) -> None:
        if self._released:
            return
        self.ctx.release()
        self._released = True


# Back-compat alias used by older imports / demos
build_procedural_lot = build_static_lot
