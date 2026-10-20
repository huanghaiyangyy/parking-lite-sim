#!/usr/bin/env python3
"""Check preview pixel layout and ownership without X11, CAN or rendering."""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import numpy as np
from PIL import Image
import pygame
from view_scene import rgb_surface


def main():
    pygame.init()
    pygame.display.set_mode((1280, 800))
    try:
        rng = np.random.default_rng(19020)
        for width, height in ((1280, 800), (319, 197), (7, 5)):
            image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            for arr in (image, image[::-1, ::2], image.transpose(1, 0, 2)):
                for size in (None, (29, 17)):
                    surface = rgb_surface(arr, size)
                    expected = (arr if size is None else np.asarray(
                        Image.fromarray(np.ascontiguousarray(arr)).resize(
                            size, Image.Resampling.BILINEAR)))
                    assert surface.get_bitsize() == pygame.display.get_surface().get_bitsize()
                    assert pygame.image.tostring(surface, 'RGB') == expected.tobytes()
            surface = rgb_surface(image)
            before = pygame.image.tostring(surface, 'RGB')
            image[:] = 0
            assert pygame.image.tostring(surface, 'RGB') == before
    finally:
        pygame.quit()
    print('PASS: preview RGB pixels, odd row widths, noncontiguous inputs, resize and ownership')


if __name__ == '__main__':
    main()
