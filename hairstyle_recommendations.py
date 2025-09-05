# hairstyle_recommendations.py

def get_hairstyle_suggestions(face_shape, hair_type=None, gender=None):
    
    print(f"get_hairstyle_suggestions called with face_shape: {face_shape}, hair_type: {hair_type}, gender: {gender}")
    print(f"face_shape type: {type(face_shape)}")
    
    # Ensure face_shape is a string
    if not isinstance(face_shape, str):
        face_shape = str(face_shape)
    
    # Dictionary mapping face shapes to hairstyle suggestions with image paths
    suggestions = {
        "oval": [
            {"name": "Buzz Cut", "image": "/static/suggestions/oval/oval_face_buzz_cut.jpg"},
            {"name": "Crew Cut", "image": "/static/suggestions/oval/oval_face_crew_cut.jpg"},
            {"name": "Messy Fringe", "image": "/static/suggestions/oval/oval_face_messy_fringe.jpg"},
            {"name": "Textured Crop", "image": "/static/suggestions/oval/oval_face_textured_crop.jpg"},
            {"name": "Textured Quiff", "image": "/static/suggestions/oval/oval_face_textured_quiff.jpg"},
            {"name": "Soft Curls", "image": "/static/suggestions/oval/soft_curls.jpg"}
        ],
        "round": [
            {"name": "Pompadour", "image": "/static/suggestions/round/round-classic-pompadour.jpg"},
            {"name": "Asymmentrical hairstyle_1", "image": "/static/suggestions/round/round_asymmetrical_hairstyle.jpg"},
            {"name": "Asymmentrical hairstyle_2", "image": "/static/suggestions/round/round_asymmetrical_hairstyle_2.jpg"},
            {"name": "Buzz and beard", "image": "/static/suggestions/round/round-smooth-buzz-cut-with-beard.jpg"},
            {"name": "Spiky hairstyle", "image": "/static/suggestions/round/round_spiky_hairstyle.jpg"},
            {"name": "Vertical hairstyle", "image": "/static/suggestions/round/round-vertical-hairstyle.jpg"}
        ],
        "square": [
            {"name": "Buzz Cut with Fade", "image": "/static/suggestions/square/squre_Buzz_Cut_with_Fade.jpg"},
            {"name": "Long Hairstyles", "image": "/static/suggestions/square/squre_Long_Hairstyles.jpg"},
            {"name": "Medium shag", "image": "/static/suggestions/square/squre_Medium_Shag.jpg"},
            {"name": "Slick back", "image": "/static/suggestions/square/squre_Slick_Back.jpg"},
            {"name": "Textured Crop", "image": "/static/suggestions/square/squre_Textured_Crop.jpg"},
            {"name": "Tousled Waves", "image": "/static/suggestions/square/squre_Tousled_Waves.jpg"}
        ],
        "heart": [
            {"name": "Backe Swept Hairstyle", "image": "/static/suggestions/heart/heart_Backe_Swept_Hairstyle.jpg"},
            {"name": "Lenth fringes", "image": "/static/suggestions/heart/heart_Lenth_fringes.jpg"},
            {"name": "Messy Texture", "image": "/static/suggestions/heart/heart_Messy_Texture.jpg"},
            {"name": "Fringes side swept", "image": "/static/suggestions/heart/heart_fringes_side_swept.jpg"},
            {"name": "Glamorous quiff", "image": "/static/suggestions/heart/heart_glamorous_quiff.jpg"},
            {"name": "Side Part", "image": "/static/suggestions/heart/heart_side_part.jpg"}
        ],
        "diamond": [
            {"name": "Slick Back Undercut", "image": "/static/suggestions/diamond/diamoand_slick_back_undercut.jpg"},
            {"name": "Caesar Cut", "image": "/static/suggestions/diamond/diamond_caesar_cut.jpg"},
            {"name": "Crew Cut", "image": "/static/suggestions/diamond/diamond_crew_cut.jpg"},
            {"name": "Fring", "image": "/static/suggestions/diamond/diamond_fring.jpg"},
            {"name": "High Fade With Long Top", "image": "/static/suggestions/diamond/diamond_high_fade_with_long_top.jpg"},
            {"name": "Low Fade with Textured Top", "image": "/static/suggestions/diamond/diamond_low_fade_with_textured_top.jpg"}
        ],
        "oblong": [
            {"name": "Brush up", "image": "/static/suggestions/oblong/oblong_brush_up.jpg"},
            {"name": "Faux Hawk", "image": "/static/suggestions/oblong/oblong_faux_hawk.jpg"},
            {"name": "Lvy Leauge", "image": "/static/suggestions/oblong/oblong_lvy_league.jpg"},
            {"name": "Modern Spikes", "image": "/static/suggestions/oblong/oblong_modern_spikes.jpg"},
            {"name": "Lonside Part", "image": "/static/suggestions/oblong/oblong_long_side_part.jpg"},
            {"name": "Under Cut", "image": "/static/suggestions/oblong/oblong_under_cut.jpg"}
        ]
    }

    # Debug: Log the face_shape after lowering
    print(f"face_shape after lowering: {face_shape.lower()}")

    # Default case if face shape not found
    base_suggestions = suggestions.get(face_shape.lower(), [{"name": "General Style", "image": "/static/suggestions/default/general_style.jpg"}])
    print(f"base_suggestions after lookup: {base_suggestions}")

    # Apply hair_type modifications
    if hair_type and isinstance(hair_type, str):
        print(f"Applying hair_type modification: {hair_type}")
        if hair_type.lower() == "curly":
            base_suggestions = [{"name": f" {s['name']}", "image": s["image"]} for s in base_suggestions]
        elif hair_type.lower() == "straight":
            base_suggestions = [{"name": f"Straight {s['name']}", "image": s["image"]} for s in base_suggestions]
        elif hair_type.lower() == "wavy":
            base_suggestions = [{"name": f"Wavy {s['name']}", "image": s["image"]} for s in base_suggestions]
        print(f"base_suggestions after hair_type modification: {base_suggestions}")

    # Apply gender modifications
    if gender and isinstance(gender, str):
        print(f"Applying gender modification: {gender}")
        if gender.lower() == "male":
            base_suggestions = [{"name": f"Men's {s['name']}", "image": s["image"]} for s in base_suggestions]
        print(f"base_suggestions after gender modification: {base_suggestions}")

    # Debug: Log the return value
    print(f"Returning base_suggestions: {base_suggestions}")
    
    return base_suggestions

if __name__ == "__main__":
    # Example usage for testing
    print(get_hairstyle_suggestions("done"))