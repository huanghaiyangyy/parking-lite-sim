"""ModernGL render backend: static lot + dynamic ego + OpenCV fisheye / pinhole."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import glm
import moderngl
import numpy as np

from scene.ego_vehicle import build_ego_mesh
from scene.lot import build_static_lot, surface_material_ids
from sensors.fisheye_opencv import build_fisheye_from_pinhole_maps

ROOT = Path(__file__).resolve().parents[1]


SCENE_VS = """
#version 330 core
in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;
in float in_material;
uniform mat4 u_mvp;
uniform mat4 u_model;
out vec3 v_normal;
out vec3 v_color;
out vec3 v_world;
flat out int v_material;
void main() {
    vec4 world = u_model * vec4(in_position, 1.0);
    v_world = world.xyz;
    v_normal = mat3(u_model) * in_normal;
    v_color = in_color;
    v_material = int(in_material);
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

SCENE_FS = """
#version 330 core
in vec3 v_normal;
in vec3 v_color;
in vec3 v_world;
flat in int v_material;
uniform vec3 u_light_dir;
uniform vec3 u_cam_pos;
uniform bool u_realistic;
uniform bool u_shadowed;
uniform bool u_use_capture_mask;
uniform sampler2D u_capture_mask;
uniform sampler2D u_surface_noise;
uniform sampler2DShadow u_shadow;
uniform mat4 u_light_vp;
out vec4 f_color;

// World-space metres keep the pattern fixed as cameras/ego move; mipmaps
// filter the texture and derivatives filter joints at grazing fisheye angles.
vec3 surface_color(vec3 n) {
    if (v_material == 0)
        return v_color;
    vec2 uv = abs(n.y) > 0.65 ? v_world.xz
        : vec2(abs(n.x) > abs(n.z) ? v_world.z : v_world.x, v_world.y);
    float grain = texture(u_surface_noise, uv * 2.0).r - 0.5;
    float mottling = texture(u_surface_noise, uv * 0.047).g - 0.5;
    if (v_material == 2) {
        vec3 color = vec3(0.70, 0.685, 0.65) * (1.0 + 0.06*grain + 0.18*mottling);
        if (abs(n.y) < 0.65) {
            // Cast concrete panels: 2.4 m wide, 1.2 m high, 12 mm joints.
            vec2 spacing = vec2(2.4, 1.2);
            vec2 edge = min(mod(uv, spacing), spacing - mod(uv, spacing));
            vec2 joint = 1.0 - smoothstep(vec2(0.006),
                vec2(0.006) + max(fwidth(uv), vec2(0.001)), edge);
            color *= 1.0 - 0.18 * max(joint.x, joint.y);
            float damp = 1.0 - smoothstep(0.05, 0.48 + 0.25*mottling, v_world.y);
            color *= 1.0 - 0.22*damp;
        }
        return color;
    }
    if (v_material == 1)
        return vec3(0.225, 0.235, 0.245) * (1.0 + 0.22*grain + 0.14*mottling);
    if (v_material == 3)
        return v_color * (1.0 + 0.09*grain + 0.12*mottling);
    if (v_material == 5) {
        // Light concrete top, darker vertical face and grime at road contact.
        // Existing geometry already supplies the 16.5 cm riser and bevel.
        vec3 color = vec3(0.69, 0.68, 0.64) * (0.90 + 0.10*abs(n.y));
        float along = abs(n.x) > abs(n.z) ? v_world.z : v_world.x;
        float seam = abs(fract(along) - 0.5);
        float joint = 1.0 - smoothstep(0.004, 0.004 + max(fwidth(along), 0.001), seam);
        color *= 1.0 - 0.15*joint;
        return color * (1.0 + 0.10*grain + 0.10*mottling);
    }
    if (v_material == 6)
        return v_color * (0.97 + 0.05*grain + 0.05*mottling);
    if (v_material == 7)
        return v_color * (0.98 + 0.03*grain);
    if (v_material == 4)
        return v_color * (0.94 + 0.06*grain + 0.05*mottling);
    return v_color;
}

float visibility(vec3 n) {
    vec4 p = u_light_vp * vec4(v_world, 1.0);
    vec3 q = p.xyz / p.w * 0.5 + 0.5;
    if (any(lessThan(q, vec3(0.0))) || any(greaterThan(q, vec3(1.0))))
        return 1.0;
    float bias = max(0.00015, 0.0006 * (1.0 - max(dot(n, normalize(-u_light_dir)), 0.0)));
    vec2 step = 0.75 / vec2(textureSize(u_shadow, 0));
    return 0.25 * (
        texture(u_shadow, vec3(q.xy + vec2(-step.x, -step.y), q.z - bias)) +
        texture(u_shadow, vec3(q.xy + vec2( step.x, -step.y), q.z - bias)) +
        texture(u_shadow, vec3(q.xy + vec2(-step.x,  step.y), q.z - bias)) +
        texture(u_shadow, vec3(q.xy + vec2( step.x,  step.y), q.z - bias)));
}

void main() {
    if (u_use_capture_mask && texelFetch(u_capture_mask, ivec2(gl_FragCoord.xy), 0).r < 0.5)
        discard;
    vec3 n = normalize(v_normal);
    float ndl = max(dot(n, normalize(-u_light_dir)), 0.0);
    float fog = 0.0;
    vec3 lit;
    if (u_realistic) {
        // Simple lighting keeps surface orientation cues without shadow-map,
        // specular or fog work. The previous lighting remains opt-in.
        float sky = 0.62 + 0.08*max(n.y, 0.0);
        float shadow = u_shadowed && ndl > 0.0 ? visibility(n) : 1.0;
        lit = surface_color(n) * (sky + 0.40*ndl*shadow);
        if (u_shadowed) {
            if (v_material == 6 || v_material == 7) {
                vec3 view_dir = normalize(u_cam_pos - v_world);
                float sheen = pow(max(dot(reflect(normalize(u_light_dir), n), view_dir), 0.0), 32.0);
                lit += vec3(v_material == 7 ? 0.16 : 0.06) * sheen * shadow;
            }
            fog = clamp((length(v_world - u_cam_pos) - 25.0) / 130.0, 0.0, 0.25);
        }
    } else {
        lit = v_color * (0.42 + 0.78*ndl);
        fog = clamp((length(v_world - u_cam_pos) - 8.0) / 40.0, 0.0, 0.55);
    }
    vec3 fog_col = vec3(0.28, 0.30, 0.33);
    f_color = vec4(mix(lit, fog_col, fog), 1.0);
}
"""

SHADOW_VS = """
#version 330 core
in vec3 in_position;
uniform mat4 u_mvp;
void main() { gl_Position = u_mvp * vec4(in_position, 1.0); }
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

# Reuse the CPU Kannala–Brandt LUT. Match OpenCV INTER_LINEAR's 1/32-pixel
# coordinates and integer rounding, including black invalid samples.
KB_FISHEYE_FS = """
#version 330 core
in vec2 v_uv;
uniform sampler2D u_tex;
uniform sampler2D u_map;
out vec4 f_color;
ivec3 pixel(ivec2 p) {
    ivec2 size = textureSize(u_tex, 0);
    p = clamp(p, ivec2(0), size - 1);
    p.y = size.y - 1 - p.y;
    return ivec3(round(texelFetch(u_tex, p, 0).rgb * 255.0));
}
void main() {
    ivec2 size = textureSize(u_map, 0);
    ivec2 output_pixel = ivec2(v_uv * vec2(size));
    output_pixel.y = size.y - 1 - output_pixel.y;
    vec2 p = texelFetch(u_map, output_pixel, 0).rg;
    if (p.x < 0.0 || p.y < 0.0) {
        f_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    ivec2 q = ivec2(roundEven(p * 32.0));
    ivec2 base = q / 32;
    ivec2 f = q - base * 32;
    ivec3 value = pixel(base) * (32-f.x) * (32-f.y)
        + pixel(base+ivec2(1,0)) * f.x * (32-f.y)
        + pixel(base+ivec2(0,1)) * (32-f.x) * f.y
        + pixel(base+ivec2(1,1)) * f.x * f.y;
    f_color = vec4(vec3((value + 512) / 1024) / 255.0, 1.0);
}
"""

GRID_PREVIEW_FS = """
#version 330 core
uniform sampler2D u_tex;
out vec4 f_color;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy) * 2;
    ivec3 sum = ivec3(round(texelFetch(u_tex, p, 0).rgb * 255.0))
        + ivec3(round(texelFetch(u_tex, p+ivec2(1,0), 0).rgb * 255.0))
        + ivec3(round(texelFetch(u_tex, p+ivec2(0,1), 0).rgb * 255.0))
        + ivec3(round(texelFetch(u_tex, p+ivec2(1,1), 0).rgb * 255.0));
    f_color = vec4(vec3((sum+2)/4) / 255.0, 1.0);
}
"""

def capture_sample_mask(mx, my, width, height):
    """KB source footprint plus complete 2x2 quads for texture derivatives."""
    valid = (mx >= 0.0) & (my >= 0.0)
    x = np.rint(mx[valid] * 32.0).astype(np.int32) // 32
    y = np.rint(my[valid] * 32.0).astype(np.int32) // 32
    mask = np.zeros((height + height % 2, width + width % 2), dtype=np.uint8)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        mask[np.clip(y + dy, 0, height - 1), np.clip(x + dx, 0, width - 1)] = 255
    # A partial quad discarded before fwidth/texture can change mip selection
    # or joint filtering. Keep all four invocations for every required quad.
    quads = mask.reshape(mask.shape[0] // 2, 2, mask.shape[1] // 2, 2)
    quads[:] = quads.max(axis=(1, 3), keepdims=True)
    return np.ascontiguousarray(mask[:height, :width][::-1])


def _mat4_bytes(m: glm.mat4) -> bytes:
    return np.array(m, dtype="f4").T.tobytes()


def look_at_view(eye, target, up) -> glm.mat4:
    """View matrix that keeps vehicle-left on the left of the image.

    glm.lookAt's side vector is forward × up. With this scene's +Z-forward
    cameras that puts vehicle-left on the right side of the framebuffer.
    Negate that row. Callers draw with front face CW so culling still matches.
    """
    view = glm.lookAt(glm.vec3(*eye), glm.vec3(*target), glm.vec3(*up))
    for col in range(4):
        view[col][0] *= -1
    return view


def flu_ego_model(x: float, y: float, z: float, yaw_rad: float) -> glm.mat4:
    """Model matrix: local GL ego mesh → world GL from world-FLU pose."""
    t = glm.vec3(-y, z, x)
    return glm.translate(t) * glm.rotate(-float(yaw_rad), glm.vec3(0.0, 1.0, 0.0))


class ModernGLFisheyeBackend:
    """Offscreen renderer with static lot VAO + dynamic ego VAO.

    Default fisheye path: wide pinhole FBO → OpenCV Kannala–Brandt
    LUT (config K/D) → GPU remap. Legacy equidistant GPU remap still available
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
        pinhole_capture_scale: int = 4,
        material_style: str = "realistic",
        pinhole_sample_mask: bool = True,
        lighting_style: str = "simple",
    ):
        if material_style not in ("flat", "realistic"):
            raise ValueError("material_style must be 'flat' or 'realistic'")
        self.material_style = material_style
        if lighting_style not in ("simple", "shadowed"):
            raise ValueError("lighting_style must be 'simple' or 'shadowed'")
        self.lighting_style = lighting_style
        self.pinhole_sample_mask = bool(pinhole_sample_mask)
        self.width = int(width)
        self.height = int(height)
        if pinhole_capture_scale not in (1, 2, 4):
            raise ValueError("pinhole_capture_scale must be 1, 2, or 4")
        self.capture_width = self.width * pinhole_capture_scale
        self.capture_height = self.height * pinhole_capture_scale
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
        rear_oh = vcfg.get("rear_overhang_m")
        origin_fwd = 0.0
        if rear_oh is not None:
            origin_fwd = length_m * 0.5 - float(rear_oh)
        wheelbase = vcfg.get("wheelbase_m")
        wheelbase_m = float(wheelbase) if wheelbase is not None else None

        # Offscreen workers have no X display; use EGL without trying GLX.
        kwargs = {} if os.environ.get("DISPLAY") else {"backend": "egl"}
        try:
            self.ctx = moderngl.create_context(standalone=True, **kwargs)
        except Exception:
            # Keep the other backend as a fallback if the preferred one fails.
            kwargs = {"backend": "egl"} if not kwargs else {}
            self.ctx = moderngl.create_context(standalone=True, **kwargs)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        # look_at_view reflects X; clockwise faces are the outside.
        self.ctx.front_face = "cw"

        self._prog = self.ctx.program(vertex_shader=SCENE_VS, fragment_shader=SCENE_FS)

        lot_data = build_static_lot(self.assets_root).reshape(-1, 9)
        ego_data = build_ego_mesh(
            length_m,
            width_m,
            height_m,
            self.assets_root,
            origin_forward_offset_m=origin_fwd,
            wheelbase_m=wheelbase_m,
        ).reshape(-1, 9)

        # Keep the on-disk Nx9 cache unchanged; material IDs are GPU-only.
        lot_gpu = np.column_stack((lot_data, surface_material_ids(lot_data)))
        ego_gpu = np.column_stack((ego_data, np.zeros(len(ego_data), dtype=np.float32)))
        self._lot_vbo = self.ctx.buffer(lot_gpu.tobytes())
        self._lot_vao = self.ctx.vertex_array(
            self._prog,
            [(self._lot_vbo, "3f 3f 3f 1f", "in_position", "in_normal", "in_color", "in_material")],
        )
        self._ego_vbo = self.ctx.buffer(ego_gpu.tobytes())
        self._ego_vao = self.ctx.vertex_array(
            self._prog,
            [(self._ego_vbo, "3f 3f 3f 1f", "in_position", "in_normal", "in_color", "in_material")],
        )

        noise = np.random.default_rng(20261019).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        self._surface_noise = self.ctx.texture((256, 256), 3, noise.tobytes())
        self._surface_noise.build_mipmaps()
        self._surface_noise.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self._shadow_tex = self.ctx.depth_texture((1536, 1536))
        self._shadow_tex.compare_func = "<="
        self._shadow_tex.repeat_x = self._shadow_tex.repeat_y = False
        self._shadow_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._shadow_fbo = self.ctx.framebuffer(depth_attachment=self._shadow_tex)
        self._shadow_prog = self.ctx.program(vertex_shader=SHADOW_VS,
                                              fragment_shader="#version 330 core\nvoid main() {}")
        self._shadow_vaos = [self.ctx.vertex_array(self._shadow_prog,
            [(vbo, "3f 28x", "in_position")]) for vbo in (self._lot_vbo, self._ego_vbo)]
        self._shadow_dirty = True

        self._color_tex = self.ctx.texture((self.width, self.height), 3)
        depth_rb = self.ctx.depth_renderbuffer((self.width, self.height))
        self._fbo = self.ctx.framebuffer(
            color_attachments=[self._color_tex], depth_attachment=depth_rb
        )
        self._capture_fbo = None
        self._capture_tex = None

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
        self._kb_prog = self.ctx.program(vertex_shader=FISHEYE_VS, fragment_shader=KB_FISHEYE_FS)
        self._kb_vao = self.ctx.vertex_array(
            self._kb_prog, [(quad_vbo, "2f 2f", "in_vert", "in_uv")]
        )
        self._kb_maps = {}
        self._capture_masks = {}
        self._grid_fbo = None
        self._preview_fbo = None
        self._quad_vbo = quad_vbo
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
        model = flu_ego_model(x, y, z, yaw_rad)
        if model != self._ego_model:
            self._shadow_dirty = True
        self._ego_model = model

    def _update_shadow(self) -> None:
        if (not self._shadow_dirty or self.material_style == "flat"
                or self.lighting_style != "shadowed"):
            return
        center = glm.vec3(self._ego_model[3])
        light = center - glm.normalize(glm.vec3(*self._light_dir)) * 60.0
        self._light_vp = glm.ortho(-24.0, 24.0, -24.0, 24.0, 0.1, 120.0) * glm.lookAt(
            light, center, glm.vec3(0.0, 1.0, 0.0))
        self._shadow_fbo.use()
        self.ctx.viewport = (0, 0, *self._shadow_fbo.size)
        self._shadow_fbo.clear(depth=1.0)
        # The light view has no X reflection, unlike the camera view.
        self.ctx.front_face = "ccw"
        self.ctx.polygon_offset = (1.1, 2.0)
        try:
            for model, vao in zip((self._lot_model, self._ego_model), self._shadow_vaos):
                self._shadow_prog["u_mvp"].write(_mat4_bytes(self._light_vp * model))
                vao.render(moderngl.TRIANGLES)
        finally:
            self.ctx.polygon_offset = (0.0, 0.0)
            self.ctx.front_face = "cw"
        self._shadow_dirty = False

    def _ensure_capture(self) -> None:
        if self._capture_fbo is None:
            if (self.capture_width, self.capture_height) == (self.width, self.height):
                self._capture_fbo, self._capture_tex = self._fbo, self._color_tex
            else:
                size = (self.capture_width, self.capture_height)
                self._capture_tex = self.ctx.texture(size, 3)
                self._capture_fbo = self.ctx.framebuffer(color_attachments=[self._capture_tex],
                                                       depth_attachment=self.ctx.depth_renderbuffer(size))
                self._capture_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                self._capture_tex.repeat_x = self._capture_tex.repeat_y = False

    def _draw_scene(self, proj: glm.mat4, view: glm.mat4, eye: Sequence[float], fbo=None,
                    sample_mask=None) -> None:
        self._update_shadow()
        target = self._fbo if fbo is None else fbo
        target.use()
        self.ctx.viewport = (0, 0, *target.size)
        self.ctx.clear(0.28, 0.30, 0.33, 1.0)
        self._prog["u_use_capture_mask"].value = sample_mask is not None
        self._prog["u_capture_mask"].value = 4
        if sample_mask is not None:
            sample_mask.use(location=4)
        self._prog["u_light_dir"].value = self._light_dir
        self._prog["u_cam_pos"].value = (float(eye[0]), float(eye[1]), float(eye[2]))
        self._prog["u_realistic"].value = self.material_style == "realistic"
        self._prog["u_shadowed"].value = self.lighting_style == "shadowed"
        self._surface_noise.use(location=2)
        self._shadow_tex.use(location=3)
        self._prog["u_surface_noise"].value = 2
        self._prog["u_shadow"].value = 3
        if self.material_style == "realistic" and self.lighting_style == "shadowed":
            self._prog["u_light_vp"].write(_mat4_bytes(self._light_vp))

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
        width, height = fbo.size
        return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)[::-1].copy()

    def _read_fbo_rgba(self, fbo) -> np.ndarray:
        """Bottom-up RGBA as stored by GL. The Jetson helper flips rows."""
        data = fbo.read(components=4, alignment=1)
        width, height = fbo.size
        return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)

    def _render_wide_pinhole(
        self,
        eye: Sequence[float],
        target: Sequence[float],
        up: Sequence[float],
    ) -> np.ndarray:
        view = look_at_view(eye, target, up)
        self._ensure_capture()
        self._draw_scene(self._proj_fisheye_pinhole, view, eye, self._capture_fbo)
        return self._read_fbo_rgb(self._capture_fbo)

    def _kb_lut(self, K, D):
        if self.fisheye_mode in ("opencv", "cv2", "kannala", "kb") and K is not None and D is not None:
            key = (self.pinhole_fov_y_deg, self.capture_width, self.capture_height,
                   np.asarray(K, dtype=np.float64).tobytes(),
                   np.asarray(D, dtype=np.float64).tobytes())
            lut = self._kb_maps.get(key)
            if lut is None:
                mx, my = build_fisheye_from_pinhole_maps(
                    self.width, self.height, K, D, self.pinhole_fov_y_deg,
                    self.capture_width, self.capture_height)
                lut = self.ctx.texture((self.width, self.height), 2,
                    np.stack((mx, my), axis=-1).tobytes(), dtype="f4")
                self._kb_maps[key] = lut
                mask = capture_sample_mask(mx, my, self.capture_width, self.capture_height)
                self._capture_masks[key] = self.ctx.texture(
                    (self.capture_width, self.capture_height), 1, mask.tobytes())
            return lut, self._capture_masks[key]
        return None, None

    def _remap_gpu(self, K, D, output, viewport) -> None:
        self._ensure_capture()
        output.use()
        self.ctx.viewport = viewport
        lut, _ = self._kb_lut(K, D)
        if lut is not None:
            lut.use(location=1)
            self._kb_prog["u_tex"].value = 0
            self._kb_prog["u_map"].value = 1
            vao = self._kb_vao
        else:
            self._fish_prog["u_tex"].value = 0
            self._fish_prog["u_pinhole_fov_y"].value = math.radians(self.pinhole_fov_y_deg)
            self._fish_prog["u_fisheye_fov"].value = self.fisheye_fov
            self._fish_prog["u_aspect"].value = self.aspect
            vao = self._fish_vao
        self._capture_tex.use(location=0)
        # Scene faces are CW; the fullscreen quad is CCW.
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

    def _render_camera_to_fbo(self, eye, target, up, K, D, output, viewport) -> None:
        if self._released:
            raise RuntimeError("ModernGLFisheyeBackend already released")
        self._ensure_capture()
        _, mask = self._kb_lut(K, D)
        if not self.pinhole_sample_mask or self.material_style == "flat":
            mask = None
        self._draw_scene(self._proj_fisheye_pinhole, look_at_view(eye, target, up), eye,
                         self._capture_fbo, sample_mask=mask)
        self._remap_gpu(K, D, output, viewport)

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
        self._render_camera_to_fbo(eye, target, up, K, D, self._fish_fbo,
                                   (0, 0, self.width, self.height))
        return self._read_fbo_rgb(self._fish_fbo)

    def render_grid(self, cameras) -> np.ndarray:
        """Render four (eye, target, up, K, D) views into a 2x2 atlas; read once."""
        if self._released:
            raise RuntimeError("ModernGLFisheyeBackend already released")
        if len(cameras) != 4:
            raise ValueError("camera grid requires exactly four views")
        if self._grid_fbo is None:
            texture = self.ctx.texture((self.width*2, self.height*2), 3)
            self._grid_fbo = self.ctx.framebuffer(color_attachments=[texture])
        for i, (eye, target, up, K, D) in enumerate(cameras):
            viewport = ((i%2)*self.width, (1-i//2)*self.height, self.width, self.height)
            self._render_camera_to_fbo(eye, target, up, K, D, self._grid_fbo, viewport)
        return self._read_fbo_rgb(self._grid_fbo)

    def render_grid_rgba(self, cameras) -> np.ndarray:
        """Same atlas as render_grid, read back once as bottom-up RGBA."""
        if self._released:
            raise RuntimeError("ModernGLFisheyeBackend already released")
        if len(cameras) != 4:
            raise ValueError("camera grid requires exactly four views")
        if self._grid_fbo is None:
            texture = self.ctx.texture((self.width * 2, self.height * 2), 3)
            self._grid_fbo = self.ctx.framebuffer(color_attachments=[texture])
        for i, (eye, target, up, K, D) in enumerate(cameras):
            viewport = ((i % 2) * self.width, (1 - i // 2) * self.height, self.width, self.height)
            self._render_camera_to_fbo(eye, target, up, K, D, self._grid_fbo, viewport)
        return self._read_fbo_rgba(self._grid_fbo)

    def draw_grid(self, cameras) -> None:
        """Draw the 2x2 atlas and return before the readback."""
        if self._released:
            raise RuntimeError("ModernGLFisheyeBackend already released")
        if len(cameras) != 4:
            raise ValueError("camera grid requires exactly four views")
        if self._grid_fbo is None:
            texture = self.ctx.texture((self.width * 2, self.height * 2), 3)
            self._grid_fbo = self.ctx.framebuffer(color_attachments=[texture])
        for i, (eye, target, up, K, D) in enumerate(cameras):
            viewport = ((i % 2) * self.width, (1 - i // 2) * self.height, self.width, self.height)
            self._render_camera_to_fbo(eye, target, up, K, D, self._grid_fbo, viewport)

    def read_grid_rgba(self) -> np.ndarray:
        if self._grid_fbo is None:
            raise RuntimeError("camera grid has not been drawn")
        return self._read_fbo_rgba(self._grid_fbo)

    def read_grid_preview(self) -> np.ndarray:
        """Half-size RGB preview of the same atlas, with exact 2x2 averaging."""
        if self._grid_fbo is None:
            raise RuntimeError("camera grid has not been drawn")
        if self._preview_fbo is None:
            tex = self.ctx.texture((self.width, self.height), 4)
            self._preview_fbo = self.ctx.framebuffer(color_attachments=[tex])
            self._preview_prog = self.ctx.program(vertex_shader=FISHEYE_VS,
                                                  fragment_shader=GRID_PREVIEW_FS)
            self._preview_vao = self.ctx.vertex_array(
                self._preview_prog, [(self._quad_vbo, "2f 8x", "in_vert")])
        self._preview_fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self._grid_fbo.color_attachments[0].use(location=0)
        self._preview_prog["u_tex"].value = 0
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self._preview_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        return self._read_fbo_rgb(self._preview_fbo)

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
        view = look_at_view(eye, target, up)
        self._draw_scene(proj, view, eye)
        return self._read_fbo_rgb(self._fbo)

    def release(self) -> None:
        if self._released:
            return
        self.ctx.release()
        self._released = True


# Back-compat alias used by older imports / demos
build_procedural_lot = build_static_lot
