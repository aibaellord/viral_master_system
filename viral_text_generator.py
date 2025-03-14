import random
import re
import numpy as np
import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ViralTextGenerator:
    """
    A specialized class for generating viral optimized text content.
    This class provides methods for analyzing content structures, generating
    compelling hooks, body content, and calls to action, and optimizing text
    for maximum virality across platforms.
    """
    
    # Common viral hook patterns
    HOOK_PATTERNS = {
        "question": "Have you ever wondered {}?",
        "statistic": "{}% of people don't know this one simple truth about {}.",
        "controversy": "Everything you know about {} is wrong.",
        "secret": "The secret to {} that experts don't want you to know.",
        "story": "I never thought {} until this happened...",
        "challenge": "I tried {} for {} days. Here's what happened.",
        "list": "{} ways to {} that will change your life.",
        "how_to": "How to {} in just {} minutes a day.",
        "prediction": "This is why {} will change everything in {}.",
        "fear": "Warning: {} could be destroying your {}."
    }
    
    # Common viral CTA patterns
    CTA_PATTERNS = {
        "engagement": "Let me know in the comments if you've experienced this too.",
        "share": "Share this with someone who needs to see this.",
        "action": "Try this today and let me know what you think.",
        "follow": "Follow for more insights on {}.",
        "subscribe": "Subscribe to never miss new content about {}.",
        "download": "Download my free guide on {} (link in bio).",
        "purchase": "Get my comprehensive program on {} (limited time offer).",
        "tag": "Tag someone who needs to know this about {}.",
        "save": "Save this post for when you need it later.",
        "question": "What's your experience with {}? Drop it below!"
    }
    
    # Emotional triggers that drive viral content
    EMOTIONAL_TRIGGERS = {
        "awe": ["breathtaking", "mind-blowing", "phenomenal", "spectacular", "extraordinary"],
        "surprise": ["shocking", "unexpected", "astonishing", "remarkable", "revolutionary"],
        "fear": ["alarming", "critical", "urgent", "crucial", "vital"],
        "joy": ["delightful", "exciting", "joyful", "thrilling", "amazing"],
        "anger": ["outrageous", "infuriating", "unjust", "maddening", "frustrating"],
        "empathy": ["moving", "touching", "heart-wrenching", "compassionate", "relatable"],
        "curiosity": ["mysterious", "intriguing", "fascinating", "curious", "wonder"],
        "validation": ["proven", "scientific", "expert-approved", "research-backed", "validated"],
        "inspiration": ["motivational", "inspiring", "empowering", "transformative", "life-changing"],
        "determination": ["unstoppable", "persistent", "resolute", "unwavering", "driven"],
        "hope": ["promising", "hopeful", "optimistic", "encouraging", "reassuring"],
        "satisfaction": ["rewarding", "gratifying", "fulfilling", "satisfying", "worthwhile"]
    }
    
    # Viral text structures by platform
    PLATFORM_STRUCTURES = {
        "twitter": {"max_length": 280, "hook_importance": 0.7, "hashtags": True},
        "instagram": {"max_length": 2200, "hook_importance": 0.6, "hashtags": True},
        "facebook": {"max_length": 5000, "hook_importance": 0.5, "hashtags": False},
        "linkedin": {"max_length": 3000, "hook_importance": 0.4, "hashtags": True},
        "tiktok": {"max_length": 2200, "hook_importance": 0.9, "hashtags": True},
        "blog": {"max_length": 50000, "hook_importance": 0.6, "hashtags": False},
        "email": {"max_length": 10000, "hook_importance": 0.8, "hashtags": False}
    }
    
    def __init__(self, nlp_processor=None, trend_analyzer=None, sentiment_analyzer=None):
        """
        Initialize the Viral Text Generator with optional external processors.
        
        Args:
            nlp_processor: Natural language processing component
            trend_analyzer: Component for analyzing current content trends
            sentiment_analyzer: Component for analyzing emotional impact
        """
        self.nlp_processor = nlp_processor
        self.trend_analyzer = trend_analyzer
        self.sentiment_analyzer = sentiment_analyzer
        logger.info("ViralTextGenerator initialized")
        
    def generate_viral_text(self, 
                           topic: str, 
                           platform: str, 
                           target_audience: Dict[str, Any] = None,
                           target_emotions: List[str] = None,
                           keywords: List[str] = None,
                           max_length: int = None,
                           style: str = "conversational") -> Dict[str, Any]:
        """
        Generate viral-optimized text content based on provided parameters.
        
        Args:
            topic: The main subject of the content
            platform: Target platform (twitter, instagram, facebook, etc.)
            target_audience: Dictionary with audience demographics and preferences
            target_emotions: List of emotions to evoke (from EMOTIONAL_TRIGGERS)
            keywords: List of keywords to include for SEO/discoverability
            max_length: Maximum character length (defaults to platform standard)
            style: Writing style (conversational, professional, etc.)
            
        Returns:
            Dictionary containing the generated text, metrics, and component parts
        """
        logger.info(f"Generating viral text for topic: {topic} on platform: {platform}")
        
        # Set defaults if not provided
        target_audience = target_audience or {"age": "25-34", "interests": [topic]}
        target_emotions = target_emotions or ["curiosity", "surprise"]
        keywords = keywords or [topic]
        max_length = max_length or self.PLATFORM_STRUCTURES.get(platform, {}).get("max_length", 1000)
        
        # Analyze the optimal structure for this content
        structure = self.analyze_optimal_structure(topic, platform, target_audience, target_emotions)
        
        # Perform keyword analysis and optimization
        keyword_data = self.analyze_keywords(keywords, topic, platform)
        
        # Create emotional journey map
        emotion_map = self.create_emotional_journey(target_emotions, structure)
        
        # Generate main content components
        hook = self.generate_hook(topic, platform, emotion_map[0], keyword_data, target_audience)
        body_content = self.generate_body_content(topic, structure, emotion_map[1:-1], keyword_data, target_audience)
        cta = self.generate_cta(topic, platform, emotion_map[-1], target_audience)
        
        # Assemble the final text
        generated_text = self.assemble_text(hook, body_content, cta, structure)
        
        # Optimize for the platform
        optimized_text = self.optimize_for_platform(generated_text, platform, max_length, style)
        
        # Calculate virality metrics
        virality_metrics = self.calculate_virality_metrics(optimized_text, platform, target_audience, target_emotions)
        
        result = {
            "text": optimized_text,
            "components": {
                "hook": hook,
                "body": body_content,
                "cta": cta
            },
            "metrics": virality_metrics,
            "platform": platform,
            "target_emotions": target_emotions,
            "optimal_hashtags": self.generate_hashtags(topic, keywords) if self.PLATFORM_STRUCTURES.get(platform, {}).get("hashtags", False) else []
        }
        
        logger.info(f"Generated viral text of length {len(optimized_text)} characters")
        return result
    
    def analyze_optimal_structure(self, topic: str, platform: str, audience: Dict[str, Any], emotions: List[str]) -> Dict[str, Any]:
        """
        Analyze and determine the optimal content structure for viral potential.
        
        Args:
            topic: Content topic
            platform: Target platform
            audience: Target audience information
            emotions: Target emotions to evoke
            
        Returns:
            Structure configuration dictionary with section arrangements and emphasis
        """
        logger.debug(f"Analyzing optimal structure for {topic} on {platform}")
        
        # Get platform-specific base structure
        platform_structure = self.PLATFORM_STRUCTURES.get(platform, self.PLATFORM_STRUCTURES["facebook"])
        
        # Determine section distribution based on platform, topic and emotions
        if "curiosity" in emotions or "surprise" in emotions:
            hook_weight = platform_structure["hook_importance"] + 0.1
        else:
            hook_weight = platform_structure["hook_importance"]
            
        # Determine optimal number of paragraphs/sections based on platform
        if platform in ["twitter", "instagram"]:
            num_sections = random.randint(2, 4)
        elif platform in ["facebook", "linkedin"]:
            num_sections = random.randint(3, 6)
        else:
            num_sections = random.randint(5, 10)
            
        # Determine if bullets/lists would be effective for this content
        use_bullets = random.random() < 0.7  # 70% chance to use bullets for most viral content
        
        # Determine pattern emphasis (e.g., question-answer, problem-solution)
        if "fear" in emotions:
            primary_pattern = "problem-solution"
        elif "curiosity" in emotions:
            primary_pattern = "mystery-reveal"
        else:
            primary_pattern = "narrative-insight"
            
        structure = {
            "hook_weight": hook_weight,
            "num_sections": num_sections,
            "use_bullets": use_bullets,
            "primary_pattern": primary_pattern,
            "section_arrangement": self._generate_section_arrangement(num_sections, primary_pattern),
            "suggested_length": min(platform_structure["max_length"], 
                                   self._calculate_optimal_length(platform, audience))
        }
        
        return structure
    
    def _generate_section_arrangement(self, num_sections: int, pattern: str) -> List[str]:
        """
        Generate the arrangement of content sections based on the pattern.
        
        Args:
            num_sections: Number of content sections
            pattern: Primary content pattern
            
        Returns:
            List of section types in optimal order
        """
        sections = []
        
        if pattern == "problem-solution":
            sections = ["problem", "impact", "solution", "evidence", "implementation", "benefits"]
        elif pattern == "mystery-reveal":
            sections = ["mystery", "clue", "background", "reveal", "explanation", "application"]
        elif pattern == "narrative-insight":
            sections = ["story", "relatability", "challenge", "turning_point", "lesson", "application"]
        
        # Trim or extend to match requested number of sections
        if len(sections) > num_sections:
            sections = sections[:num_sections]
        elif len(sections) < num_sections:
            # Repeat some sections for longer content
            additional = ["example", "elaboration", "reinforcement", "analogy"]
            while len(sections) < num_sections:
                sections.append(additional[len(sections) % len(additional)])
                
        return sections
    
    def _calculate_optimal_length(self, platform: str, audience: Dict[str, Any]) -> int:
        """Calculate the optimal content length based on platform and audience"""
        base_length = self.PLATFORM_STRUCTURES.get(platform, {}).get("max_length", 1000)
        
        # Adjust based on audience age (younger = shorter, older = longer)
        age_factor = 1.0
        age_group = audience.get("age", "25-34")
        if age_group in ["13-17", "18-24"]:
            age_factor = 0.7
        elif age_group in ["35-44", "45-54"]:
            age_factor = 1.2
        elif age_group in ["55-64", "65+"]:
            age_factor = 1.3
            
        return int(base_length * age_factor)
    
    def analyze_keywords(self, keywords: List[str], topic: str, platform: str) -> Dict[str, Any]:
        """
        Analyze keywords for viral potential and optimization.
        
        Args:
            keywords: List of target keywords
            topic: Main content topic
            platform: Target platform
            
        Returns:
            Dictionary with keyword analysis data
        """
        logger.debug(f"Analyzing keywords: {keywords}")
        
        # If we had a real trend analyzer, we would use it here
        if self.trend_analyzer:
            trending_score = self.trend_analyzer.analyze_keywords(keywords, platform)
        else:
            # Simulate keyword analysis
            trending_score = {k: random.uniform(0.1, 0.9) for k in keywords}
            
        # Identify primary and secondary keywords
        sorted_keywords = sorted(trending_score.items(), key=lambda x: x[1], reverse=True)
        primary_keywords = [k for k, s in sorted_keywords[:2]]
        secondary_keywords = [k for k, s in sorted_keywords[2:]]
        
        # Generate related keywords
        related_keywords = self._generate_related_keywords(primary_keywords, topic)
        
        return {
            "primary_keywords": primary_keywords,
            "secondary_keywords": secondary_keywords,
            "related_keywords": related_keywords,
            "trending_scores": trending_score,
            "recommended_density": self._calculate_keyword_density(platform)
        }
    
    def _generate_related_keywords(self, keywords: List[str], topic: str) -> List[str]:
        """Generate related keywords based on primary keywords and topic"""
        # In a real implementation, this would use ML/NLP to find related terms
        related = []
        for keyword in keywords:
            # Simulate related keyword generation
            related.extend([
                f"{keyword} tips",
                f"best {keyword}",
                f"{keyword} for {topic}",
                f"how to {keyword}",
                f"{keyword} examples"
            ])
        return related
    
    def _calculate_keyword_density(self, platform: str) -> float:
        """Calculate recommended keyword density based on platform"""
        densities = {
            "blog": 0.02,  # 2%
            "facebook": 0.015,
            "instagram": 0.02,
            "twitter": 0.03,
            "linkedin": 0.018,
            "email": 0.015
        }
        return densities.get(platform, 0.02)
    
    def create_emotional_journey(self, target_emotions: List[str], structure: Dict[str, Any]) -> List[str]:
        """
        Create an emotional journey map for the content progression.
        
        Args:
            target_emotions: List of target emotions to evoke
            structure: Content structure details
            
        Returns:
            List of emotions in sequence for content progression
        """
        logger.debug(f"Creating emotional journey with emotions: {target_emotions}")
        
        # Ensure we have enough emotions to work with
        all_emotions = target_emotions.copy()
        while len(all_emotions) < structure["num_sections"] + 2:  # +2 for hook and CTA
            additional = random.choice(list(self.EMOTIONAL_TRIGGERS.keys()))
            if additional not in all_emotions:
                all_emotions.append(additional)
        
        # Map the emotional journey based on the pattern
        journey = []
        pattern = structure["primary_pattern"]
        
        if pattern == "problem-solution":
            # Start with negative emotions, end with positive
            if "fear" in all_emotions:
                journey.append("fear")
            elif "anger" in all_emotions:
                journey.append("anger")
            else:
                journey.append("curiosity")
                
            # Middle emotions
            middle_emotions = []
            if "empathy" in all_emotions:
                middle_emotions.append("empathy")
            middle_emotions.extend([e for e in all_emotions if e not in ["fear", "anger", "joy", "empathy", "curiosity"]])
            
            # Add enough middle emotions to match structure
            while len(journey) + 1 < structure["num_sections"] + 1:  # +1 because we need space for final emotion
                if middle_emotions:
                    journey.append(middle_emotions.pop(0))
                else:
                    journey.append("surprise")
            
            # End with positive emotion
            if "joy" in all_emotions:
                journey.append("joy")
            else:
                journey.append("validation")
                
        elif pattern == "mystery-reveal":
            # Start with curiosity, build through surprise, end with awe
            journey.append("curiosity")
            
            # Middle emotions - mixture of curiosity, surprise
            if "surprise" in all_emotions:
                next_emotion = "surprise"
            else:
                next_emotion = "curiosity"
            journey.append(next_emotion)
            
            middle_emotions = [e for e in all_emotions if e not in ["curiosity", "surprise", "awe"]]
            while len(journey) + 1 < structure["num_sections"] + 1:
                if middle_emotions:
                    journey.append(middle_emotions.pop(0))
                else:
                    journey.append("curiosity" if journey[-1] != "curiosity" else "surprise")
            
            # End with awe or validation
            if "awe" in all_emotions:
                journey.append("awe")
            else:
                journey.append("validation")
                
        elif pattern == "narrative-insight":
            # Start with relatable emotion, journey through challenge, end with inspiration
            if "empathy" in all_emotions:
                journey.append("empathy")
            else:
                journey.append("curiosity")
                
            # Include some challenging emotion
            if "fear" in all_emotions:
                journey.append("fear")
            elif "anger" in all_emotions:
                journey.append("anger")
            else:
                journey.append("surprise")
                
            # Middle emotions
            middle_emotions = [e for e in all_emotions if e not in [journey[0], journey[1], "joy", "validation"]]
            while len(journey) + 1 < structure["num_sections"] + 1:
                if middle_emotions:
                    journey.append(middle_emotions.pop(0))
                else:
                    journey.append("curiosity")
            
            # End with positive emotion
            if "joy" in all_emotions:
                journey.append("joy")
            else:
                journey.append("validation")
        else:
            # Default journey: curiosity → various emotions → validation
            journey.append("curiosity")
            
            available_emotions = [e for e in all_emotions if e != "curiosity"]
            while len(journey) + 1 < structure["num_sections"] + 1:
                if available_emotions:
                    journey.append(available_emotions.pop(0))
                else:
                    journey.append(random.choice(list(self.EMOTIONAL_TRIGGERS.keys())))
            
            journey.append("validation")
            
        # Ensure the journey has exactly the right number of emotions
        if len(journey) > structure["num_sections"] + 2:
            journey = journey[:structure["num_sections"] + 2]
            
        logger.debug(f"Created emotional journey: {journey}")
        return journey
    
    def generate_hook(self, topic: str, platform: str, emotion: str, keyword_data: Dict[str, Any], audience: Dict[str, Any]) -> str:
        """
        Generate a compelling hook for the content.
        
        Args:
            topic: Content topic
            platform: Target platform
            emotion: Emotion to evoke in the hook
            keyword_data: Keyword analysis data
            audience: Target audience information
            
        Returns:
            Hook text string
        """
        logger.debug(f"Generating hook for {topic} with emotion: {emotion}")
        
        # Determine optimal hook type based on platform and emotion
        hook_types = list(self.HOOK_PATTERNS.keys())
        
        # Emphasize different hooks based on emotion
        emotion_to_hook_map = {
            "curiosity": ["question", "secret", "mystery"],
            "fear": ["warning", "controversy", "statistic"],
            "surprise": ["statistic", "controversy", "secret"],
            "awe": ["statistic", "prediction", "secret"],
            "joy": ["story", "list", "how_to"],
            "anger": ["controversy", "question", "statistic"],
            "empathy": ["story", "question", "challenge"],
            "validation": ["statistic", "list", "how_to"]
        }
        
        preferred_hooks = emotion_to_hook_map.get(emotion, hook_types)
        
        # Platform-specific considerations
        if platform == "twitter":
            # Shorter hooks for Twitter
            preferred_hooks = [h for h in preferred_hooks if h in ["question", "statistic", "secret"]]
        elif platform == "linkedin":
            # More professional hooks for LinkedIn
            preferred_hooks = [h for h in preferred_hooks if h in ["statistic", "how_to", "list"]]
        
        hook_type = random.choice(preferred_hooks[:2]) if preferred_hooks else random.choice(hook_types)
        
        # Get template and fill it
        template = self.HOOK_PATTERNS[hook_type]
        
        # Get keywords to incorporate
        primary_keywords = keyword_data["primary_keywords"]
        
        # Generate hook parameters based on hook type
        params = {}
        if hook_type == "question":
            params = {"": f"why {topic} is changing {random.choice(['everything', 'the industry', 'your life'])}"}
        elif hook_type == "statistic":
            params = {
                "": str(random.randint(67, 94)),  # High numbers are more shocking
                "": topic
            }
        elif hook_type == "controversy":
            params = {"": topic}
        elif hook_type == "secret":
            params = {"": topic}
        elif hook_type == "story":
            params = {"": f"I could {topic}"}
        elif hook_type == "challenge":
            params = {
                "": topic,
                "": str(random.choice([7, 14, 21, 30]))
            }
        elif hook_type == "list":
            params = {
                "": str(random.choice([3, 5, 7, 10])),
                "": topic
            }
        elif hook_type == "how_to":
            params = {
                "": topic,
                "": str(random.choice([5, 10, 15, 20]))
            }
        elif hook_type == "prediction":
            params = {
                "": topic,
                "": str(random.choice(["2024", "the next decade", "the future"]))
            }
        elif hook_type == "fear":
            params = {
                "": topic,
                "": random.choice(["health", "business", "relationships", "future"])
            }
        
        # Insert parameters into template
        hook = template.format(*params.values())
        
        # Add emotional trigger words
        emotional_words = self.EMOTIONAL_TRIGGERS.get(emotion, [])
        if emotional_words:
            trigger_word = random.choice(emotional_words)
            # Add the emotional trigger if it's not already in the hook
            if trigger_word.lower() not in hook.lower():
                # 50% chance to add at beginning, 50% chance to add within
                if random.random() < 0.5:
                    hook = f"{trigger_word.capitalize()}: {hook}"
                else:
                    words = hook.split()
                    insert_position = min(3, len(words))
                    words.insert(insert_position, trigger_word)
                    hook = " ".join(words)
        
        # Ensure a primary keyword is included
        if primary_keywords and not any(k.lower() in hook.lower() for k in primary_keywords):
            hook = f"{hook} {primary_keywords[0].capitalize()}."
            
        logger.debug(f"Generated hook: {hook}")
        return hook
    
    def generate_body_content(self, topic: str, structure: Dict[str, Any], emotions: List[str], 
                             keyword_data: Dict[str, Any], audience: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate the main body content based on structure and emotional journey.
        
        Args:
            topic: Content topic
            structure: Content structure
            emotions: List of emotions for sections
            keyword_data: Keyword data
            audience: Target audience information
            
        Returns:
            List of body content sections with text and type
        """
        logger.debug(f"Generating body content for {topic} with {len(emotions)} sections")
        
        body_sections = []
        section_types = structure["section_arrangement"]
        
        # Ensure we don't try to use more emotions than we have
        if len(emotions) < len(section_types):
            # Repeat emotions if needed
            emotions = emotions + emotions
            emotions = emotions[:len(section_types)]
            
        # Generate each section
        for i, (section_type, emotion) in enumerate(zip(section_types, emotions)):
            # Get keywords to emphasize in this section
            if i < len(keyword_data["primary_keywords"]):
                emphasis_keyword = keyword_data["primary_keywords"][i]
            elif keyword_data["secondary_keywords"]:
                emphasis_keyword = random.choice(keyword_data["secondary_keywords"])
            else:
                emphasis_keyword = topic
                
            # Generate paragraph based on section type
            if section_type == "problem":
                content = f"Many people struggle with {topic}. {random.choice(self.EMOTIONAL_TRIGGERS[emotion])} challenges arise when {emphasis_keyword} isn't properly understood."
                
            elif section_type == "impact":
                content = f"The impact of this {emphasis_keyword} issue can be {random.choice(self.EMOTIONAL_TRIGGERS[emotion])}. It affects everything from your daily routine to long-term success."
                
            elif section_type == "solution":
                content = f"The solution to this {emphasis_keyword} challenge is surprisingly simple. By implementing a {random.choice(self.EMOTIONAL_TRIGGERS[emotion])} approach, you can overcome these obstacles."
                
            elif section_type == "evidence":
                content = f"Research shows that proper {emphasis_keyword} techniques are {random.choice(self.EMOTIONAL_TRIGGERS[emotion])}. Studies have demonstrated a significant improvement in results."
                
            elif section_type == "implementation":
                content = f"Implementing this {emphasis_keyword} strategy requires just three steps. First, identify your specific needs. Second, apply the {random.choice(self.EMOTIONAL_TRIGGERS[emotion])} technique. Third, measure and adjust your approach."
                
            elif section_type == "benefits":
                content = f"The benefits of mastering {emphasis_keyword} are truly {random.choice(self.EMOTIONAL_TRIGGERS[emotion])}. You'll experience improved results, reduced stress, and new opportunities."
                
            elif section_type == "mystery":
                content = f"There's a hidden aspect of {emphasis_keyword} that few people recognize. This {random.choice(self.EMOTIONAL_TRIGGERS[emotion])} secret changes everything about how we approach {topic}."
                
            elif section_type == "clue":
                content = f"The first clue comes from observing how {emphasis_keyword} operates in optimal conditions. Notice the {random.choice(self.EMOTIONAL_TRIGGERS[emotion])} pattern that emerges."
                
            elif section_type == "background":
                content = f"To understand this {emphasis_keyword} phenomenon, we need to look at its origins. The history behind this is actually quite {random.choice(self.EMOTIONAL_TRIGGERS[emotion])}."
                
            elif section_type == "reveal":
                content = f"Here's the {random.choice(self.EMOTIONAL_TRIGGERS[emotion])} truth about {emphasis_keyword}: it's not what most people think. The reality transforms how you'll approach {topic} forever."
                
            elif section_type == "explanation":
                content = f"This works because of a fundamental principle in {emphasis_keyword}. The mechanism is surprisingly {random.choice(self.EMOTIONAL_TRIGGERS[emotion])} once you understand it."
                
            elif section_type == "application":
                content = f"Applying this to your own {emphasis_keyword} situation is straightforward. The process is {random.choice(self.EMOTIONAL_TRIGGERS[emotion])} in its effectiveness."
                
            elif section_type == "story":
                content = f"Let me share a quick story about {emphasis_keyword}. When I first encountered this challenge, the situation seemed impossible. What happened next was truly {random.choice(self.EMOTIONAL_TRIGGERS[emotion])}."
                
            elif section_type == "relatability":
                content = f"You've probably experienced similar {emphasis_keyword} challenges yourself. That feeling of uncertainty is {random.choice(self.EMOTIONAL_TRIGGERS[emotion])} and something we all face."
                
            elif section_type == "challenge":
                content = f"The real challenge with {emphasis_keyword} isn't what most think. It's actually the {random.choice(self.EMOTIONAL_TRIGGERS[emotion])} obstacle that few discuss."
                
            elif section_type == "turning_point":
                content = f"The turning point comes when you realize this key truth about {emphasis_keyword}. This {random.choice(self.EMOTIONAL_TRIGGERS[emotion])} insight changes everything."
                
            elif section_type == "lesson":
                content = f"The lesson here is {random.choice(self.EMOTIONAL_TRIGGERS[emotion])}: mastering {emphasis_keyword} isn't about working harder but working differently."
                
            else:  # For any other section types
                content = f"When considering {emphasis_keyword}, remember that the approach needs to be {random.choice(self.EMOTIONAL_TRIGGERS[emotion])}. This principle applies universally to {topic}."
                
            # Add to sections
            body_sections.append({
                "type": section_type,
                "content": content,
                "emotion": emotion
            })
            
        # Add bullets if structure suggests it
        if structure["use_bullets"] and len(body_sections) > 2:
            # Identify a section to convert to bullets
            bullet_section_index = random.randint(1, len(body_sections) - 1)
            bullet_section = body_sections[bullet_section_index]
            
            # Create bullet points
            bullet_content = [
                f"• {emphasis_keyword} optimization: Enhance your approach with targeted techniques",
                f"• Consistent application: Apply {random.choice(self.EMOTIONAL_TRIGGERS[bullet_section['emotion']])} methods daily",
                f"• Measurement: Track results and adjust your {emphasis_keyword} strategy accordingly",
                f"• Refinement: Continuously improve your {topic} approach based on feedback"
            ]
            
            # Replace the section with bullet points
            body_sections[bullet_section_index]["content"] = "\n".join([
                f"Key points to remember about {emphasis_keyword}:",
                *bullet_content
            ])
            
        logger.debug(f"Generated {len(body_sections)} body sections")
        return body_sections
    
    def generate_cta(self, topic: str, platform: str, emotion: str, audience: Dict[str, Any]) -> str:
        """
        Generate a compelling call-to-action for the content.
        
        Args:
            topic: Content topic
            platform: Target platform
            emotion: Emotion to evoke in the CTA
            audience: Target audience information
            
        Returns:
            CTA text string
        """
        logger.debug(f"Generating CTA for {topic} on {platform} with emotion: {emotion}")
        
        # Select CTA type based on platform and audience
        cta_types = list(self.CTA_PATTERNS.keys())
        
        # Platform-specific CTAs
        platform_cta_map = {
            "instagram": ["share", "tag", "save", "follow"],
            "twitter": ["engagement", "share", "follow"],
            "facebook": ["engagement", "share", "tag"],
            "linkedin": ["engagement", "follow", "subscribe"],
            "tiktok": ["follow", "share", "engagement"],
            "blog": ["subscribe", "download", "action"],
            "email": ["action", "purchase", "download"]
        }
        
        preferred_ctas = platform_cta_map.get(platform, cta_types)
        
        # Emotion-based CTA selection
        if emotion in ["joy", "awe"]:
            # Positive emotions work best with sharing
            if "share" in preferred_ctas:
                preferred_ctas = [c for c in preferred_ctas if c in ["share", "tag"]] + preferred_ctas
        elif emotion in ["curiosity", "surprise"]:
            # Curious emotions work well with engagement
            if "engagement" in preferred_ctas:
                preferred_ctas = [c for c in preferred_ctas if c in ["engagement", "question"]] + preferred_ctas
        
        cta_type = random.choice(preferred_ctas[:2]) if preferred_ctas else random.choice(cta_types)
        
        # Get template and fill it
        template = self.CTA_PATTERNS[cta_type]
        
        # Generate parameters for the template
        params = {}
        if "{}" in template:
            params = {"": topic}
        
        # Insert parameters into template
        cta = template.format(*params.values())
        
        # Add emotional trigger
        emotional_words = self.EMOTIONAL_TRIGGERS.get(emotion, [])
        if emotional_words and random.random() < 0.7:  # 70% chance to add emotional trigger
            trigger_word = random.choice(emotional_words)
            if trigger_word.lower() not in cta.lower():
                cta = f"{cta} It will be {trigger_word}!"
        
        logger.debug(f"Generated CTA: {cta}")
        return cta
    
    def assemble_text(self, hook: str, body_sections: List[Dict[str, str]], cta: str, structure: Dict[str, Any]) -> str:
        """
        Assemble the final text from hook, body content, and CTA.
        
        Args:
            hook: The hook text
            body_sections: List of body content sections
            cta: The call-to-action text
            structure: Content structure details
            
        Returns:
            Assembled text content
        """
        logger.debug("Assembling final text content")
        
        # Start with the hook
        text_parts = [hook]
        
        # Add body sections
        for section in body_sections:
            text_parts.append(section["content"])
        
        # Add CTA
        text_parts.append(cta)
        
        # Join with proper spacing based on platform
        if structure.get("platform") in ["twitter", "instagram"]:
            # These platforms often use line breaks
            assembled_text = "\n\n".join(text_parts)
        else:
            # More traditional paragraph format
            assembled_text = "\n\n".join(text_parts)
            
        logger.debug(f"Assembled text of length {len(assembled_text)}")
        return assembled_text
    
    def optimize_for_platform(self, text: str, platform: str, max_length: int, style: str) -> str:
        """
        Optimize the text for specific platform requirements and constraints.
        
        Args:
            text: The text to optimize
            platform: Target platform
            max_length: Maximum allowed length
            style: Content style
            
        Returns:
            Optimized text
        """
        logger.debug(f"Optimizing text for {platform} with max length {max_length}")
        
        optimized_text = text
        
        # Apply platform-specific formatting
        if platform == "twitter":
            # Twitter needs shorter content
            if len(optimized_text) > max_length:
                # Trim body paragraphs while preserving hook and CTA
                paragraphs = optimized_text.split("\n\n")
                if len(paragraphs) > 2:
                    hook = paragraphs[0]
                    cta = paragraphs[-1]
                    body = paragraphs[1:-1]
                    
                    # Remove paragraphs until within limit
                    while len("\n\n".join([hook] + body + [cta])) > max_length and len(body) > 1:
                        body.pop()
                    
                    optimized_text = "\n\n".join([hook] + body + [cta])
                
                # Final truncation if still too long
                if len(optimized_text) > max_length:
                    optimized_text = optimized_text[:max_length-3] + "..."
        
        elif platform == "instagram":
            # Instagram often uses emojis and more line breaks
            # Add some line breaks for readability
            optimized_text = re.sub(r'(\n\n)', '\n\n\n', optimized_text)
            
            # Add some emojis based on content
            if "list" in optimized_text.lower():
                optimized_text = optimized_text.replace("•", "✅")
            
            if len(optimized_text) > max_length:
                optimized_text = optimized_text[:max_length-3] + "..."
        
        elif platform in ["facebook", "linkedin"]:
            # More professional style for LinkedIn
            if platform == "linkedin" and style == "professional":
                # Remove any overly casual language
                optimized_text = re.sub(r'(!{2,})', '!', optimized_text)
                
            if len(optimized_text) > max_length:
                # Trim while preserving structure
                paragraphs = optimized_text.split("\n\n")
                while len("\n\n".join(paragraphs)) > max_length and len(paragraphs) > 2:
                    # Remove a paragraph from the middle
                    mid_point = len(paragraphs) // 2
                    paragraphs.pop(mid_point)
                
                optimized_text = "\n\n".join(paragraphs)
                
                # Final truncation if still too long
                if len(optimized_text) > max_length:
                    optimized_text = optimized_text[:max_length-3] + "..."
        
        logger.debug(f"Optimized text length: {len(optimized_text)}")
        return optimized_text
    
    def calculate_virality_metrics(self, text: str, platform: str, audience: Dict[str, Any], target_emotions: List[str]) -> Dict[str, float]:
        """
        Calculate predicted virality metrics for the generated content.
        
        Args:
            text: The generated text
            platform: Target platform
            audience: Target audience information
            target_emotions: Target emotions
            
        Returns:
            Dictionary with virality metrics
        """
        logger.debug(f"Calculating virality metrics for content on {platform}")
        
        metrics = {}
        
        # Calculate engagement score (0-100)
        # In a real implementation, this would use ML models trained on viral content
        
        # Basic text quality factors
        text_length = len(text)
        num_paragraphs = text.count("\n\n") + 1
        
        # Calculate base score from length and structure
        length_score = min(100, (text_length / 1000) * 100) if platform != "twitter" else min(100, (text_length / 280) * 100)
        structure_score = min(100, num_paragraphs * 10)  # More paragraphs typically better for readability
        
        # Check for emotional trigger words
        emotion_words_count = 0
        for emotion in target_emotions:
            for trigger_word in self.EMOTIONAL_TRIGGERS.get(emotion, []):
                emotion_words_count += text.lower().count(trigger_word.lower())
        
        emotion_score = min(100, emotion_words_count * 15)
        
        # Platform-specific factors
        platform_factors = {
            "twitter": {"length": 0.2, "emotion": 0.5, "structure": 0.3},
            "instagram": {"length": 0.3, "emotion": 0.5, "structure": 0.2},
            "facebook": {"length": 0.4, "emotion": 0.3, "structure": 0.3},
            "linkedin": {"length": 0.5, "emotion": 0.2, "structure": 0.3},
            "tiktok": {"length": 0.1, "emotion": 0.7, "structure": 0.2},
            "blog": {"length": 0.6, "emotion": 0.2, "structure": 0.2},
            "email": {"length": 0.4, "emotion": 0.4, "structure": 0.2}
        }
        
        factors = platform_factors.get(platform, {"length": 0.33, "emotion": 0.34, "structure": 0.33})
        
        # Calculate final engagement score
        engagement_score = (
            (length_score * factors["length"]) +
            (emotion_score * factors["emotion"]) +
            (structure_score * factors["structure"])
        )
        
        # Round to 2 decimal places
        engagement_score = round(engagement_score, 2)
        
        # Additional metrics
        metrics["engagement_score"] = engagement_score
        metrics["estimated_reach"] = round(engagement_score * 1.2, 2)
        metrics["shareability"] = round(min(100, engagement_score * random.uniform(0.8, 1.2)), 2)
        metrics["conversion_potential"] = round(engagement_score * 0.7, 2)
        
        # Audience match score
        audience_age_group = audience.get("age", "25-34")
        if audience_age_group in ["18-24", "25-34"]:
            if platform in ["instagram", "tiktok"]:
                metrics["audience_match"] = round(min(100, engagement_score * 1.1), 2)
            else:
                metrics["audience_match"] = round(min(100, engagement_score * 0.9), 2)
        elif audience_age_group in ["35-44", "45-54"]:
            if platform in ["facebook", "linkedin"]:
                metrics["audience_match"] = round(min(100, engagement_score * 1.1), 2)
            else:
                metrics["audience_match"] = round(min(100, engagement_score * 0.9), 2)
        else:
            metrics["audience_match"] = round(engagement_score, 2)
        
        logger.debug(f"Calculated virality metrics: engagement_score={engagement_score}")
        return metrics
    
    def generate_hashtags(self, topic: str, keywords: List[str], max_tags: int = 10) -> List[str]:
        """
        Generate optimized hashtags for the content.
        
        Args:
            topic: Content topic
            keywords: List of keywords
            max_tags: Maximum number of hashtags to generate
            
        Returns:
            List of hashtag strings
        """
        logger.debug(f"Generating hashtags for {topic}")
        
        hashtags = []
        
        # Add topic as main hashtag
        main_tag = "#" + "".join(topic.title().split())
        hashtags.append(main_tag)
        
        # Add keyword-based hashtags
        for keyword in keywords:
            # Remove special characters and spaces
            clean_keyword = "".join(e for e in keyword if e.isalnum() or e == " ")
            tag = "#" + "".join(clean_keyword.title().split())
            if tag not in hashtags:
                hashtags.append(tag)
        
        # Add some general popular hashtags based on topic
        general_tags = {
            "marketing": ["#MarketingTips", "#DigitalMarketing", "#ContentStrategy"],
            "business": ["#Entrepreneur", "#BusinessTips", "#Success"],
            "fitness": ["#FitnessJourney", "#HealthyLifestyle", "#WorkoutMotivation"],
            "food": ["#FoodLover", "#Recipes", "#FoodPhotography"],
            "travel": ["#TravelGram", "#Wanderlust", "#TravelPhotography"],
            "technology": ["#Tech", "#Innovation", "#DigitalTransformation"],
            "fashion": ["#StyleInspiration", "#FashionTrends", "#OOTD"],
            "education": ["#Learning", "#Education", "#StudyTips"],
            "finance": ["#FinancialFreedom", "#MoneyTips", "#Investment"],
            "health": ["#HealthTips", "#Wellness", "#SelfCare"]
        }
        
        # Try to find matching category or use topic as category
        matching_category = None
        for category, tags in general_tags.items():
            if category.lower() in topic.lower():
                matching_category = category
                break
        
        if matching_category:
            for tag in general_tags[matching_category]:
                if tag not in hashtags:
                    hashtags.append(tag)
        
        # Add trending tags (simulated here)
        trending_tags = ["#Trending", f"#{datetime.datetime.now().year}", "#MustRead"]
        for tag in trending_tags:
            if tag not in hashtags and random.random() < 0.3:  # 30% chance to include each
                hashtags.append(tag)
        
        # Trim to max number
        if len(hashtags) > max_tags:
            hashtags = hashtags[:max_tags]
            
        logger.debug(f"Generated {len(hashtags)} hashtags")
        return hashtags
