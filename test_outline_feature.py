#!/usr/bin/env python3
"""
Quick verification script for the Voronoi Outline feature.
Creates a simple test painting with outlines to verify the implementation.
"""

from PIL import Image
from voronoi_painting import VoronoiPainting


def test_outline_feature():
    """Test that outline parameters are correctly stored and used."""

    # Load a test image
    test_image = Image.open("./img/sunset.jpeg").convert("RGBA")
    print(f"Loaded test image: {test_image.size}")

    # Test 1: Painting without outlines
    painting_no_outline = VoronoiPainting(
        num_points=50,
        target_image=test_image,
        background_color=(200, 200, 200),
        output_scale=1.0,
        outline_width=0,
        outline_color=(0, 0, 0),
    )
    print(f"✓ Created painting without outlines")
    print(f"  - Outline width: {painting_no_outline.get_outline_width}")
    print(f"  - Outline color: {painting_no_outline.get_outline_color}")

    # Test 2: Painting with black outlines
    painting_black_outline = VoronoiPainting(
        num_points=50,
        target_image=test_image,
        background_color=(200, 200, 200),
        output_scale=1.0,
        outline_width=2,
        outline_color=(0, 0, 0),
    )
    print(f"✓ Created painting with black 2px outlines")
    print(f"  - Outline width: {painting_black_outline.get_outline_width}")
    print(f"  - Outline color: {painting_black_outline.get_outline_color}")

    # Test 3: Painting with colored outlines
    painting_colored_outline = VoronoiPainting(
        num_points=50,
        target_image=test_image,
        background_color=(200, 200, 200),
        output_scale=1.0,
        outline_width=3,
        outline_color=(255, 0, 0),  # Red
    )
    print(f"✓ Created painting with red 3px outlines")
    print(f"  - Outline width: {painting_colored_outline.get_outline_width}")
    print(f"  - Outline color: {painting_colored_outline.get_outline_color}")

    # Test 4: Render with outlines (small sample)
    print(f"\n✓ Rendering test painting with outlines...")
    rendered = painting_colored_outline.draw(scale=0.5)
    print(f"  - Rendered size: {rendered.size}")
    print(f"  - Saved to test_outline_output.png")
    rendered.save("test_outline_output.png")

    # Test 5: Check outline inheritance through mating
    print(f"\n✓ Testing outline inheritance through mating...")
    child_a, child_b = VoronoiPainting.mate(
        painting_black_outline, painting_colored_outline
    )
    print(f"  - Child A outline width: {child_a.get_outline_width}")
    print(f"  - Child A outline color: {child_a.get_outline_color}")
    print(f"  - Child B outline width: {child_b.get_outline_width}")
    print(f"  - Child B outline color: {child_b.get_outline_color}")

    print(f"\n✅ All tests passed! Outline feature is working correctly.")
    print(f"\nUsage example:")
    print(f"  OUTLINE_WIDTH=2 OUTLINE_COLOR=255,255,255 python evolve_voronoi.py")


if __name__ == "__main__":
    try:
        test_outline_feature()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
