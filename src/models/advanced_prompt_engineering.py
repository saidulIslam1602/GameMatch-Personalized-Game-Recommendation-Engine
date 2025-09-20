"""
Advanced Prompt Engineering for GameMatch
Sophisticated prompting techniques for fine-tuned gaming LLMs
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import random
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PromptStrategy(Enum):
    """Available prompting strategies"""
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot" 
    CHAIN_OF_THOUGHT = "chain_of_thought"
    CONTEXTUAL = "contextual"
    COMPARATIVE = "comparative"
    PERSONA_BASED = "persona_based"
    STRUCTURED_REASONING = "structured_reasoning"
    MULTI_TURN = "multi_turn"

class UserPersona(Enum):
    """User persona types for adaptive prompting"""
    CASUAL_GAMER = "casual"
    HARDCORE_GAMER = "hardcore"
    INDIE_ENTHUSIAST = "indie"
    AAA_LOVER = "aaa"
    RETRO_GAMER = "retro"
    COMPETITIVE_PLAYER = "competitive"
    STORY_SEEKER = "story"
    EXPLORER = "explorer"

@dataclass
class PromptContext:
    """Context information for adaptive prompting"""
    user_persona: Optional[UserPersona] = None
    gaming_history: List[str] = None
    preferences: Dict[str, Any] = None
    conversation_history: List[Dict] = None
    session_context: Dict[str, Any] = None
    difficulty_level: str = "intermediate"  # beginner, intermediate, expert

class AdvancedPromptEngineer:
    """Advanced prompt engineering system for gaming recommendations"""
    
    def __init__(self):
        self.few_shot_examples = self._build_few_shot_examples()
        self.chain_of_thought_templates = self._build_cot_templates()
        self.persona_templates = self._build_persona_templates()
        self.structured_frameworks = self._build_structured_frameworks()
        
    def generate_prompt(
        self, 
        query: str, 
        strategy: PromptStrategy,
        context: PromptContext = None,
        rag_context: str = None,
        additional_info: Dict = None
    ) -> str:
        """Generate optimized prompt based on strategy and context"""
        
        context = context or PromptContext()
        additional_info = additional_info or {}
        
        prompt_generators = {
            PromptStrategy.ZERO_SHOT: self._generate_zero_shot,
            PromptStrategy.FEW_SHOT: self._generate_few_shot,
            PromptStrategy.CHAIN_OF_THOUGHT: self._generate_chain_of_thought,
            PromptStrategy.CONTEXTUAL: self._generate_contextual,
            PromptStrategy.COMPARATIVE: self._generate_comparative,
            PromptStrategy.PERSONA_BASED: self._generate_persona_based,
            PromptStrategy.STRUCTURED_REASONING: self._generate_structured_reasoning,
            PromptStrategy.MULTI_TURN: self._generate_multi_turn
        }
        
        if strategy not in prompt_generators:
            strategy = PromptStrategy.CONTEXTUAL
            
        return prompt_generators[strategy](query, context, rag_context, additional_info)
    
    def _build_few_shot_examples(self) -> Dict[str, List[Dict]]:
        """Build few-shot learning examples for different scenarios"""
        
        return {
            "similarity_recommendations": [
                {
                    "user": "I love The Witcher 3. What similar games would you recommend?",
                    "assistant": {
                        "recommendations": [
                            {
                                "game": "Dragon Age: Inquisition",
                                "reasoning": "Like The Witcher 3, it features a rich fantasy world with meaningful choices, deep RPG mechanics, and compelling companion relationships. The open-world exploration and side quest quality are comparable.",
                                "match_score": 0.89
                            },
                            {
                                "game": "Red Dead Redemption 2", 
                                "reasoning": "While different in setting, it matches The Witcher 3's attention to detail, immersive world-building, and morally complex narrative choices. Both feature extensive side content that feels meaningful.",
                                "match_score": 0.85
                            }
                        ],
                        "explanation": "I focused on games that share The Witcher 3's core strengths: immersive world-building, meaningful choices, and high-quality content depth rather than just the fantasy genre."
                    }
                },
                {
                    "user": "Games like Dark Souls that are challenging but fair?",
                    "assistant": {
                        "recommendations": [
                            {
                                "game": "Hollow Knight",
                                "reasoning": "Shares Dark Souls' precise, skill-based combat where death is a learning opportunity. The interconnected world design and gradual ability unlocking mirror Dark Souls' progression philosophy.",
                                "match_score": 0.91
                            },
                            {
                                "game": "Sekiro: Shadows Die Twice",
                                "reasoning": "From the same developer (FromSoftware), it maintains the challenging-but-fair philosophy with a focus on timing and precision. The death/resurrection mechanic preserves the learning-through-failure approach.",
                                "match_score": 0.94
                            }
                        ],
                        "explanation": "I prioritized games that embody Dark Souls' core principle: difficulty that teaches rather than frustrates, where mastery comes through understanding patterns and mechanics."
                    }
                }
            ],
            
            "genre_recommendations": [
                {
                    "user": "What are some excellent indie puzzle games?",
                    "assistant": {
                        "recommendations": [
                            {
                                "game": "The Witness",
                                "reasoning": "A masterclass in puzzle design with over 600 environmental puzzles that teach through discovery rather than tutorials. Each puzzle type builds naturally on previous concepts.",
                                "innovation_factor": "Revolutionary non-verbal teaching through environmental design",
                                "match_score": 0.95
                            },
                            {
                                "game": "Baba Is You",
                                "reasoning": "Completely reinvents puzzle mechanics by making the rules themselves manipulatable. Forces players to think about puzzle-solving in entirely new ways.",
                                "innovation_factor": "Meta-puzzle mechanics that break conventional game logic",
                                "match_score": 0.92
                            }
                        ],
                        "explanation": "I selected indie puzzle games that push boundaries and offer unique mechanics rather than just polished versions of existing puzzle types."
                    }
                }
            ],
            
            "contextual_recommendations": [
                {
                    "user": "I have 2 hours to play something relaxing after a stressful day at work. I usually enjoy strategy games but need something low-pressure right now.",
                    "assistant": {
                        "recommendations": [
                            {
                                "game": "Slime Rancher",
                                "reasoning": "Combines light management elements familiar to strategy players with a cheerful, stress-free atmosphere. Progress is always positive, with no failure states or time pressure.",
                                "session_fit": "Perfect for 2-hour sessions with natural stopping points",
                                "stress_level": "Very low - purely positive progression",
                                "match_score": 0.88
                            },
                            {
                                "game": "Mini Metro",
                                "reasoning": "Satisfies strategic thinking through transit planning but removes combat and failure anxiety. The minimalist aesthetic and ambient soundtrack promote relaxation.",
                                "session_fit": "Great for short sessions, easy to pause/resume",
                                "stress_level": "Low - meditative gameplay with gentle challenge",
                                "match_score": 0.85
                            }
                        ],
                        "context_analysis": "Recognized need for strategic elements but prioritized stress reduction and session length compatibility over complexity.",
                        "explanation": "I balanced your strategy game preference with your current need for relaxation, choosing games that engage strategic thinking without creating pressure."
                    }
                }
            ]
        }
    
    def _build_cot_templates(self) -> Dict[str, str]:
        """Chain-of-thought prompting templates"""
        
        return {
            "game_analysis": """Let me analyze this step by step:

1. **Understanding the Request**: {query_analysis}
2. **Key Requirements Identification**: {requirements}
3. **Genre and Mechanic Analysis**: {mechanics_analysis}
4. **Similar Games Consideration**: {similar_games_thought}
5. **Quality and Fit Assessment**: {quality_assessment}
6. **Final Recommendation Reasoning**: {final_reasoning}

Based on this analysis, here are my recommendations:""",

            "comparative_analysis": """I'll compare potential recommendations systematically:

**Comparison Framework:**
- **Core Mechanics**: How well do the core gameplay systems match?
- **Atmosphere & Tone**: Does the overall feel align with the request?
- **Difficulty & Accessibility**: Is the challenge level appropriate?
- **Content Depth**: How much content does the player get?
- **Innovation Factor**: What unique elements does it bring?

**Candidate Analysis:**
{candidate_analysis}

**Ranking Logic:**
{ranking_logic}

**Final Selections:**""",

            "contextual_reasoning": """Let me consider the full context:

**User Context Analysis:**
- **Stated Preferences**: {stated_preferences}
- **Implied Needs**: {implied_needs}
- **Session Context**: {session_context}
- **Experience Level**: {experience_assessment}

**Environmental Factors:**
- **Platform Considerations**: {platform_factors}
- **Time Constraints**: {time_factors}
- **Social Context**: {social_factors}

**Recommendation Strategy:**
{strategy_explanation}

**Personalized Recommendations:**"""
        }
    
    def _build_persona_templates(self) -> Dict[UserPersona, Dict]:
        """Persona-based prompt templates"""
        
        return {
            UserPersona.CASUAL_GAMER: {
                "focus": "Accessibility, fun factor, easy learning curve",
                "avoid": "Steep difficulty spikes, complex mechanics, time-consuming games",
                "language_style": "Friendly, encouraging, emphasizes enjoyment",
                "template": "As someone who games for fun and relaxation, here are some great options that are easy to pick up but genuinely entertaining..."
            },
            
            UserPersona.HARDCORE_GAMER: {
                "focus": "Mechanical depth, challenge, replayability, meta-game elements",
                "avoid": "Overly simplified games, hand-holding tutorials, casual-focused design",
                "language_style": "Technical, appreciates nuanced analysis, respects expertise", 
                "template": "For a player who appreciates mechanical depth and meaningful challenge, these games offer the complexity and engagement you're looking for..."
            },
            
            UserPersona.INDIE_ENTHUSIAST: {
                "focus": "Innovation, artistic vision, unique mechanics, developer creativity",
                "avoid": "Generic AAA titles, derivative gameplay, corporate-feeling games",
                "language_style": "Appreciates artistry, values creativity over polish",
                "template": "These indie gems showcase the creative innovation and artistic vision that make independent games special..."
            },
            
            UserPersona.STORY_SEEKER: {
                "focus": "Narrative quality, character development, emotional impact, world-building",
                "avoid": "Gameplay-only focused games, weak storytelling, repetitive content",
                "language_style": "Emphasizes narrative elements, emotional resonance",
                "template": "For a player who values great storytelling, these games offer compelling narratives and memorable characters..."
            },
            
            UserPersona.COMPETITIVE_PLAYER: {
                "focus": "Skill ceiling, competitive scene, balanced mechanics, esports potential",
                "avoid": "Single-player focused games, unbalanced mechanics, casual multiplayer",
                "language_style": "Performance-focused, appreciates competitive depth",
                "template": "These games offer the competitive depth and skill expression that serious players demand..."
            }
        }
    
    def _build_structured_frameworks(self) -> Dict[str, str]:
        """Structured reasoning frameworks"""
        
        return {
            "GAME_ANALYSIS_FRAMEWORK": """
## Game Analysis Framework

### Core Mechanics Evaluation
- **Primary Gameplay Loop**: {primary_loop}
- **Secondary Systems**: {secondary_systems}
- **Progression Mechanics**: {progression}

### Player Experience Design
- **Learning Curve**: {learning_curve}
- **Difficulty Progression**: {difficulty}
- **Engagement Patterns**: {engagement}

### Quality Assessment
- **Technical Excellence**: {technical_quality}
- **Design Cohesion**: {design_quality}
- **Innovation Factor**: {innovation}

### Contextual Fit
- **Target Audience Match**: {audience_fit}
- **Session Length Suitability**: {session_fit}
- **Platform Optimization**: {platform_fit}
""",

            "RECOMMENDATION_REASONING": """
## Recommendation Reasoning

### Match Analysis
**Primary Match Factors**:
{primary_factors}

**Secondary Considerations**:
{secondary_factors}

**Potential Concerns**:
{concerns}

### Quality Justification
**Why This Game Excels**:
{excellence_factors}

**Unique Value Proposition**:
{unique_value}

### Player Benefit Prediction
**Expected Player Experience**:
{predicted_experience}

**Long-term Satisfaction Factors**:
{satisfaction_factors}
"""
        }
    
    def _generate_zero_shot(self, query: str, context: PromptContext, rag_context: str, additional_info: Dict) -> str:
        """Generate zero-shot prompt"""
        
        base_instruction = """You are an expert game recommendation AI with deep knowledge of gaming mechanics, genres, and player psychology. Provide thoughtful, personalized game recommendations with clear reasoning.

Guidelines:
- Focus on games that truly match the user's request, not just popular titles
- Explain WHY each game fits the request
- Consider gameplay mechanics, atmosphere, and player experience
- Provide diverse recommendations when appropriate
- Include both well-known and hidden gem options when relevant"""
        
        if rag_context:
            base_instruction += f"\n\nRelevant game information:\n{rag_context}"
            
        return f"{base_instruction}\n\nUser Request: {query}\n\nRecommendations:"
    
    def _generate_few_shot(self, query: str, context: PromptContext, rag_context: str, additional_info: Dict) -> str:
        """Generate few-shot learning prompt"""
        
        # Select relevant examples based on query type
        examples = []
        query_lower = query.lower()
        
        if "like" in query_lower or "similar" in query_lower:
            examples = self.few_shot_examples["similarity_recommendations"]
        elif any(genre in query_lower for genre in ["puzzle", "strategy", "rpg", "indie"]):
            examples = self.few_shot_examples["genre_recommendations"]  
        elif any(word in query_lower for word in ["relaxing", "stressful", "time", "session"]):
            examples = self.few_shot_examples["contextual_recommendations"]
        else:
            examples = random.sample(
                self.few_shot_examples["similarity_recommendations"] + 
                self.few_shot_examples["genre_recommendations"], 2
            )
        
        prompt = "You are an expert game recommendation AI. Here are examples of high-quality recommendations:\n\n"
        
        for i, example in enumerate(examples[:2], 1):
            prompt += f"Example {i}:\n"
            prompt += f"User: {example['user']}\n"
            prompt += f"Assistant: {json.dumps(example['assistant'], indent=2)}\n\n"
        
        if rag_context:
            prompt += f"Relevant game information:\n{rag_context}\n\n"
            
        prompt += f"Now provide a similar high-quality recommendation for:\nUser: {query}\nAssistant:"
        
        return prompt
    
    def _generate_chain_of_thought(self, query: str, context: PromptContext, rag_context: str, additional_info: Dict) -> str:
        """Generate chain-of-thought prompt"""
        
        # Analyze query to select appropriate CoT template
        template_key = "game_analysis"
        if "compare" in query.lower() or "vs" in query.lower():
            template_key = "comparative_analysis"
        elif context and (context.gaming_history or context.preferences):
            template_key = "contextual_reasoning"
        
        template = self.chain_of_thought_templates[template_key]
        
        # Fill in analysis prompts
        analysis_prompts = {
            "query_analysis": f"What exactly is the user asking for in: '{query}'?",
            "requirements": "What are the key requirements I need to address?",
            "mechanics_analysis": "What gameplay mechanics and genres should I consider?",
            "similar_games_thought": "What games come to mind and why?",
            "quality_assessment": "How do I ensure high-quality, relevant recommendations?",
            "final_reasoning": "What's my reasoning for the final selections?"
        }
        
        prompt = "You are an expert game recommendation AI. Think through this request step by step:\n\n"
        prompt += template.format(**analysis_prompts)
        
        if rag_context:
            prompt += f"\n\nAvailable game information:\n{rag_context}"
        
        prompt += f"\n\nUser Request: {query}"
        
        return prompt
    
    def _generate_contextual(self, query: str, context: PromptContext, rag_context: str, additional_info: Dict) -> str:
        """Generate contextual prompt with user information"""
        
        prompt = "You are an expert game recommendation AI with access to user context. Provide personalized recommendations.\n\n"
        
        if context:
            prompt += "User Context:\n"
            if context.user_persona:
                persona_info = self.persona_templates.get(context.user_persona, {})
                prompt += f"- Player Type: {context.user_persona.value} ({persona_info.get('focus', 'General gaming preferences')})\n"
            
            if context.gaming_history:
                prompt += f"- Gaming History: {', '.join(context.gaming_history)}\n"
                
            if context.preferences:
                prompt += f"- Preferences: {json.dumps(context.preferences)}\n"
                
            if context.session_context:
                prompt += f"- Session Context: {json.dumps(context.session_context)}\n"
                
            prompt += "\n"
        
        if rag_context:
            prompt += f"Relevant Games Database:\n{rag_context}\n\n"
        
        prompt += f"User Request: {query}\n\n"
        prompt += "Provide personalized recommendations that consider the user's context and preferences:"
        
        return prompt
    
    def _generate_comparative(self, query: str, context: PromptContext, rag_context: str, additional_info: Dict) -> str:
        """Generate comparative analysis prompt"""
        
        template = self.chain_of_thought_templates["comparative_analysis"]
        
        prompt = "You are an expert game recommendation AI specializing in comparative analysis.\n\n"
        
        comparison_framework = """
        **Evaluation Criteria:**
        1. **Mechanical Similarity**: How closely do the core gameplay mechanics match?
        2. **Atmospheric Match**: Does the mood, tone, and aesthetic align?
        3. **Quality Standard**: Does the game meet high quality benchmarks?
        4. **Accessibility**: Is it appropriately challenging for the user?
        5. **Innovation**: What unique elements does it contribute?
        """
        
        prompt += comparison_framework
        
        if rag_context:
            prompt += f"\n**Available Games Data:**\n{rag_context}\n"
        
        prompt += f"\n**User Request:** {query}\n\n"
        prompt += template.format(
            candidate_analysis="[Analyze 5-7 potential candidates systematically]",
            ranking_logic="[Explain ranking decisions based on criteria above]"
        )
        
        return prompt
    
    def _generate_persona_based(self, query: str, context: PromptContext, rag_context: str, additional_info: Dict) -> str:
        """Generate persona-based prompt"""
        
        persona = context.user_persona if context else UserPersona.CASUAL_GAMER
        persona_info = self.persona_templates[persona]
        
        prompt = f"You are an expert game recommendation AI specializing in {persona.value} gamers.\n\n"
        
        prompt += f"**Player Profile:**\n"
        prompt += f"- Focus Areas: {persona_info['focus']}\n"
        prompt += f"- Avoid: {persona_info['avoid']}\n"
        prompt += f"- Communication Style: {persona_info['language_style']}\n\n"
        
        if rag_context:
            prompt += f"**Game Database:**\n{rag_context}\n\n"
        
        prompt += f"**Request:** {query}\n\n"
        prompt += persona_info['template']
        
        return prompt
    
    def _generate_structured_reasoning(self, query: str, context: PromptContext, rag_context: str, additional_info: Dict) -> str:
        """Generate structured reasoning prompt"""
        
        framework = self.structured_frameworks["GAME_ANALYSIS_FRAMEWORK"]
        reasoning = self.structured_frameworks["RECOMMENDATION_REASONING"]
        
        prompt = "You are an expert game recommendation AI using structured analysis frameworks.\n\n"
        
        prompt += "Use this structured approach for each recommendation:\n"
        prompt += framework + "\n"
        prompt += reasoning + "\n"
        
        if rag_context:
            prompt += f"**Available Game Data:**\n{rag_context}\n\n"
        
        prompt += f"**User Request:** {query}\n\n"
        prompt += "Apply the structured framework to provide detailed recommendations:"
        
        return prompt
    
    def _generate_multi_turn(self, query: str, context: PromptContext, rag_context: str, additional_info: Dict) -> str:
        """Generate multi-turn conversation prompt"""
        
        prompt = "You are an expert game recommendation AI in an ongoing conversation with a user.\n\n"
        
        if context and context.conversation_history:
            prompt += "**Conversation History:**\n"
            for i, turn in enumerate(context.conversation_history[-3:], 1):  # Last 3 turns
                prompt += f"Turn {i}:\n"
                prompt += f"User: {turn.get('user', '')}\n"
                prompt += f"Assistant: {turn.get('assistant', '')}\n\n"
        
        prompt += "**Current Context:**\n"
        prompt += "- Build upon previous recommendations\n"
        prompt += "- Reference earlier conversation points\n"
        prompt += "- Avoid repeating previous suggestions unless specifically requested\n"
        prompt += "- Show continuity in reasoning\n\n"
        
        if rag_context:
            prompt += f"**Game Database:**\n{rag_context}\n\n"
        
        prompt += f"**Latest User Message:** {query}\n\n"
        prompt += "Continue the conversation with relevant, contextual recommendations:"
        
        return prompt
    
    def optimize_for_model(self, prompt: str, model_name: str = "gpt-3.5-turbo") -> str:
        """Optimize prompt for specific model"""
        
        if "gpt-3.5" in model_name.lower():
            # GPT-3.5 specific optimizations
            if len(prompt) > 3000:  # Rough token estimation
                # Compress prompt while maintaining key information
                prompt = self._compress_prompt_gpt35(prompt)
        elif "gpt-4" in model_name.lower():
            # GPT-4 can handle longer, more complex prompts
            pass
        
        return prompt
    
    def _compress_prompt_gpt35(self, prompt: str) -> str:
        """Compress prompt for GPT-3.5 token limits"""
        
        # Remove excessive whitespace
        import re
        prompt = re.sub(r'\n\s*\n', '\n\n', prompt)
        prompt = re.sub(r' +', ' ', prompt)
        
        # Truncate examples if too long
        if len(prompt) > 3500:
            # Keep only most relevant parts
            lines = prompt.split('\n')
            if len(lines) > 50:
                prompt = '\n'.join(lines[:25] + ['...'] + lines[-20:])
        
        return prompt

# Example usage and testing
def test_prompt_engineering():
    """Test the prompt engineering system"""
    
    engineer = AdvancedPromptEngineer()
    
    test_cases = [
        {
            "query": "I love The Witcher 3, what similar games would you recommend?",
            "strategy": PromptStrategy.FEW_SHOT,
            "context": PromptContext(user_persona=UserPersona.STORY_SEEKER)
        },
        {
            "query": "Best indie puzzle games for someone new to the genre?",
            "strategy": PromptStrategy.CHAIN_OF_THOUGHT,
            "context": PromptContext(user_persona=UserPersona.CASUAL_GAMER, difficulty_level="beginner")
        },
        {
            "query": "Games like Dark Souls but more accessible",
            "strategy": PromptStrategy.COMPARATIVE,
            "context": PromptContext(user_persona=UserPersona.HARDCORE_GAMER, gaming_history=["Elden Ring", "Bloodborne"])
        }
    ]
    
    print("🧠 Testing Advanced Prompt Engineering\n" + "="*50)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {case['strategy'].value}")
        print("-" * 30)
        
        prompt = engineer.generate_prompt(
            query=case["query"],
            strategy=case["strategy"],
            context=case["context"]
        )
        
        print(f"Query: {case['query']}")
        print(f"Generated Prompt Length: {len(prompt)} characters")
        print(f"First 200 chars: {prompt[:200]}...")
        
    print("\n✅ Prompt Engineering System Ready!")

if __name__ == "__main__":
    test_prompt_engineering()