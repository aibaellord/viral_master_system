#!/usr/bin/env python3
"""
test_viral_text.py - Demonstration of the ViralTextGenerator class

This script demonstrates how to use the ViralTextGenerator class to generate
viral content for different social media platforms with different configurations.
"""

import json
from viral_text_generator import ViralTextGenerator

def print_viral_content(result, platform_name):
    """Print the viral content and its metrics in a formatted way"""
    print("\n" + "=" * 80)
    print(f"VIRAL CONTENT FOR {platform_name.upper()}")
    print("=" * 80)
    
    # Print the generated text
    print("\n📝 GENERATED CONTENT:")
    print("-" * 40)
    print(result["text"])
    print("-" * 40)
    
    # Print the content components
    print("\n🧩 CONTENT COMPONENTS:")
    print(f"Hook: {result['components']['hook']}")
    print(f"CTA: {result['components']['cta']}")
    
    # Print metrics
    print("\n📊 VIRALITY METRICS:")
    for metric, value in result["metrics"].items():
        print(f"- {metric.replace('_', ' ').title()}: {value}")
    
    # Print hashtags if available
    if result.get("optimal_hashtags"):
        print("\n🏷️ HASHTAGS:")
        print(" ".join(result["optimal_hashtags"]))
    
    print("\n")

def main():
    print("VIRAL TEXT GENERATOR DEMONSTRATION")
    print("=" * 80)
    print("Demonstrating how to generate viral content for different platforms")
    
    # Create a ViralTextGenerator instance (with default configuration)
    generator = ViralTextGenerator()
    
    # Define sample topics and audiences
    topics = {
        "twitter": "social media marketing tips",
        "instagram": "sustainable fitness habits",
        "linkedin": "leadership strategies in remote work environments"
    }
    
    audiences = {
        "twitter": {
            "age": "18-24",
            "interests": ["marketing", "social media", "digital trends"]
        },
        "instagram": {
            "age": "25-34",
            "interests": ["fitness", "health", "lifestyle"]
        },
        "linkedin": {
            "age": "35-44",
            "interests": ["business", "leadership", "professional development"]
        }
    }
    
    # Generate viral content for Twitter
    twitter_result = generator.generate_viral_text(
        topic=topics["twitter"],
        platform="twitter",
        target_audience=audiences["twitter"],
        target_emotions=["curiosity", "surprise"],
        keywords=["social media", "marketing", "viral", "engagement"],
        style="conversational"
    )
    print_viral_content(twitter_result, "Twitter")
    
    # Generate viral content for Instagram
    instagram_result = generator.generate_viral_text(
        topic=topics["instagram"],
        platform="instagram",
        target_audience=audiences["instagram"],
        target_emotions=["inspiration", "joy"],
        keywords=["fitness", "sustainable", "health", "wellness"],
        style="inspirational"
    )
    print_viral_content(instagram_result, "Instagram")
    
    # Generate viral content for LinkedIn
    linkedin_result = generator.generate_viral_text(
        topic=topics["linkedin"],
        platform="linkedin",
        target_audience=audiences["linkedin"],
        target_emotions=["validation", "curiosity"],
        keywords=["leadership", "remote work", "management", "productivity"],
        style="professional"
    )
    print_viral_content(linkedin_result, "LinkedIn")
    
    # Bonus: Generate content with custom configuration
    custom_result = generator.generate_viral_text(
        topic="AI in everyday life",
        platform="facebook",
        target_audience={"age": "25-54", "interests": ["technology", "AI", "innovation"]},
        target_emotions=["awe", "surprise", "curiosity"],
        keywords=["artificial intelligence", "AI tools", "machine learning", "future technology"],
        max_length=1500,
        style="educational"
    )
    print_viral_content(custom_result, "Facebook (Custom)")

if __name__ == "__main__":
    main()

