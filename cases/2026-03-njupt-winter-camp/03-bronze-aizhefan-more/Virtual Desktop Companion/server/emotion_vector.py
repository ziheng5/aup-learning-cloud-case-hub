from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import math

from config import EmotionLabels, ActionTypes


@dataclass
class EmotionVector:
    valence: float
    arousal: float
    dominance: float
    emotion_label: str
    intensity: float
    
    def to_list(self) -> List[float]:
        return [self.valence, self.arousal, self.dominance]
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "intensity": self.intensity
        }


@dataclass
class Live2DParams:
    eye_open: float
    eye_x: float
    eye_y: float
    mouth_open: float
    mouth_form: float
    body_angle_x: float
    body_angle_y: float
    body_angle_z: float
    breath: float
    emotion: str
    motion: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "params": {
                "eye_open": self.eye_open,
                "eye_x": self.eye_x,
                "eye_y": self.eye_y,
                "mouth_open": self.mouth_open,
                "mouth_form": self.mouth_form,
                "body_angle_x": self.body_angle_x,
                "body_angle_y": self.body_angle_y,
                "body_angle_z": self.body_angle_z,
                "breath": self.breath
            },
            "emotion": self.emotion,
            "motion": self.motion
        }


class EmotionVectorGenerator:
    def __init__(self):
        self.emotion_mapping = {
            EmotionLabels.HAPPY: {
                "valence": 0.8,
                "arousal": 0.7,
                "dominance": 0.6,
            },
            EmotionLabels.SAD: {
                "valence": 0.2,
                "arousal": 0.3,
                "dominance": 0.3,
            },
            EmotionLabels.ANGRY: {
                "valence": 0.1,
                "arousal": 0.9,
                "dominance": 0.8,
            },
            EmotionLabels.NEUTRAL: {
                "valence": 0.5,
                "arousal": 0.5,
                "dominance": 0.5,
            },
            EmotionLabels.THINKING: {
                "valence": 0.5,
                "arousal": 0.4,
                "dominance": 0.4,
            },
            EmotionLabels.EXCITED: {
                "valence": 0.9,
                "arousal": 0.95,
                "dominance": 0.7,
            },
            EmotionLabels.TIRED: {
                "valence": 0.4,
                "arousal": 0.2,
                "dominance": 0.3,
            },
            EmotionLabels.ANXIOUS: {
                "valence": 0.3,
                "arousal": 0.8,
                "dominance": 0.3,
            },
        }
        
        self.emotion_transitions = {
            ("happy", "sad"): 0.3,
            ("sad", "happy"): 0.3,
            ("angry", "happy"): 0.2,
            ("neutral", "happy"): 0.5,
            ("neutral", "sad"): 0.4,
        }
        
        self.current_vector = EmotionVector(
            valence=0.5,
            arousal=0.5,
            dominance=0.5,
            emotion_label=EmotionLabels.NEUTRAL,
            intensity=0.5
        )
    
    def generate(self, emotion_data: Dict[str, Any], 
                 smooth_factor: float = 0.3) -> EmotionVector:
        target_emotion = emotion_data.get("label", EmotionLabels.NEUTRAL)
        target_valence = emotion_data.get("valence", 0.5)
        target_arousal = emotion_data.get("arousal", 0.5)
        
        base_values = self.emotion_mapping.get(target_emotion, {
            "valence": 0.5, "arousal": 0.5, "dominance": 0.5
        })
        
        intensity = math.sqrt(
            (target_valence - 0.5) ** 2 + 
            (target_arousal - 0.5) ** 2
        ) * 2
        
        target_vector = EmotionVector(
            valence=base_values["valence"] * (0.5 + target_valence * 0.5),
            arousal=base_values["arousal"] * (0.5 + target_arousal * 0.5),
            dominance=base_values["dominance"],
            emotion_label=target_emotion,
            intensity=intensity
        )
        
        smoothed_vector = self._smooth_transition(
            self.current_vector, target_vector, smooth_factor
        )
        
        self.current_vector = smoothed_vector
        return smoothed_vector
    
    def _smooth_transition(self, current: EmotionVector, 
                           target: EmotionVector, 
                           factor: float) -> EmotionVector:
        return EmotionVector(
            valence=current.valence + (target.valence - current.valence) * factor,
            arousal=current.arousal + (target.arousal - current.arousal) * factor,
            dominance=current.dominance + (target.dominance - current.dominance) * factor,
            emotion_label=target.emotion_label,
            intensity=current.intensity + (target.intensity - current.intensity) * factor
        )


class Live2DParamMapper:
    def __init__(self):
        self.emotion_to_params = {
            EmotionLabels.HAPPY: {
                "eye_open": 1.0,
                "eye_x": 0.0,
                "eye_y": 0.1,
                "mouth_open": 0.6,
                "mouth_form": 0.8,
                "body_angle_x": 0.0,
                "body_angle_y": 5.0,
                "body_angle_z": 0.0,
            },
            EmotionLabels.SAD: {
                "eye_open": 0.5,
                "eye_x": 0.0,
                "eye_y": -0.2,
                "mouth_open": 0.2,
                "mouth_form": 0.2,
                "body_angle_x": -5.0,
                "body_angle_y": -3.0,
                "body_angle_z": 0.0,
            },
            EmotionLabels.ANGRY: {
                "eye_open": 0.9,
                "eye_x": 0.0,
                "eye_y": 0.3,
                "mouth_open": 0.4,
                "mouth_form": 0.1,
                "body_angle_x": 10.0,
                "body_angle_y": 0.0,
                "body_angle_z": 0.0,
            },
            EmotionLabels.NEUTRAL: {
                "eye_open": 0.8,
                "eye_x": 0.0,
                "eye_y": 0.0,
                "mouth_open": 0.1,
                "mouth_form": 0.5,
                "body_angle_x": 0.0,
                "body_angle_y": 0.0,
                "body_angle_z": 0.0,
            },
            EmotionLabels.THINKING: {
                "eye_open": 0.6,
                "eye_x": 0.2,
                "eye_y": 0.1,
                "mouth_open": 0.1,
                "mouth_form": 0.4,
                "body_angle_x": 5.0,
                "body_angle_y": 2.0,
                "body_angle_z": 3.0,
            },
            EmotionLabels.EXCITED: {
                "eye_open": 1.0,
                "eye_x": 0.0,
                "eye_y": 0.2,
                "mouth_open": 0.8,
                "mouth_form": 0.9,
                "body_angle_x": 0.0,
                "body_angle_y": 10.0,
                "body_angle_z": 0.0,
            },
            EmotionLabels.TIRED: {
                "eye_open": 0.3,
                "eye_x": 0.0,
                "eye_y": -0.1,
                "mouth_open": 0.1,
                "mouth_form": 0.3,
                "body_angle_x": -10.0,
                "body_angle_y": -5.0,
                "body_angle_z": 0.0,
            },
            EmotionLabels.ANXIOUS: {
                "eye_open": 0.9,
                "eye_x": 0.1,
                "eye_y": 0.0,
                "mouth_open": 0.3,
                "mouth_form": 0.3,
                "body_angle_x": 0.0,
                "body_angle_y": -5.0,
                "body_angle_z": 5.0,
            },
        }
        
        self.emotion_to_motion = {
            EmotionLabels.HAPPY: "happy_idle",
            EmotionLabels.SAD: "sad_idle",
            EmotionLabels.ANGRY: "angry_idle",
            EmotionLabels.NEUTRAL: "normal_idle",
            EmotionLabels.THINKING: "thinking",
            EmotionLabels.EXCITED: "excited",
            EmotionLabels.TIRED: "sleepy",
            EmotionLabels.ANXIOUS: "nervous",
        }
        
        self.current_params = self._create_neutral_params()
        self.breath_phase = 0.0
    
    def _create_neutral_params(self) -> Live2DParams:
        return Live2DParams(
            eye_open=0.8,
            eye_x=0.0,
            eye_y=0.0,
            mouth_open=0.1,
            mouth_form=0.5,
            body_angle_x=0.0,
            body_angle_y=0.0,
            body_angle_z=0.0,
            breath=0.5,
            emotion=EmotionLabels.NEUTRAL,
            motion="normal_idle"
        )
    
    def map_to_live2d(self, emotion_vector: EmotionVector, 
                       action: str = None,
                       smooth_factor: float = 0.4) -> Live2DParams:
        emotion = emotion_vector.emotion_label
        intensity = emotion_vector.intensity
        
        target_params = self.emotion_to_params.get(emotion, self.emotion_to_params[EmotionLabels.NEUTRAL])
        target_motion = self.emotion_to_motion.get(emotion, "normal_idle")
        
        if action:
            action_motion_map = {
                ActionTypes.WAVE: "wave",
                ActionTypes.NOD: "nod",
                ActionTypes.THINK: "thinking",
                ActionTypes.SLEEP: "sleepy",
                ActionTypes.HAPPY: "happy",
                ActionTypes.SAD: "sad",
                ActionTypes.IDLE: "normal_idle",
            }
            target_motion = action_motion_map.get(action, target_motion)
        
        self.breath_phase = (self.breath_phase + 0.05) % (2 * math.pi)
        breath = 0.5 + 0.2 * math.sin(self.breath_phase)
        
        neutral_params = self.emotion_to_params[EmotionLabels.NEUTRAL]
        
        def lerp(current, target, neutral, intensity):
            blended = neutral + (target - neutral) * intensity
            return current + (blended - current) * smooth_factor
        
        new_params = Live2DParams(
            eye_open=lerp(self.current_params.eye_open, target_params["eye_open"], neutral_params["eye_open"], intensity),
            eye_x=lerp(self.current_params.eye_x, target_params["eye_x"], neutral_params["eye_x"], intensity),
            eye_y=lerp(self.current_params.eye_y, target_params["eye_y"], neutral_params["eye_y"], intensity),
            mouth_open=lerp(self.current_params.mouth_open, target_params["mouth_open"], neutral_params["mouth_open"], intensity),
            mouth_form=lerp(self.current_params.mouth_form, target_params["mouth_form"], neutral_params["mouth_form"], intensity),
            body_angle_x=lerp(self.current_params.body_angle_x, target_params["body_angle_x"], neutral_params["body_angle_x"], intensity),
            body_angle_y=lerp(self.current_params.body_angle_y, target_params["body_angle_y"], neutral_params["body_angle_y"], intensity),
            body_angle_z=lerp(self.current_params.body_angle_z, target_params["body_angle_z"], neutral_params["body_angle_z"], intensity),
            breath=breath,
            emotion=emotion,
            motion=target_motion
        )
        
        self.current_params = new_params
        return new_params


class EmotionSystem:
    def __init__(self):
        self.vector_generator = EmotionVectorGenerator()
        self.live2d_mapper = Live2DParamMapper()
    
    def process(self, emotion_data: Dict[str, Any], 
                action: str = None) -> Dict[str, Any]:
        vector = self.vector_generator.generate(emotion_data)
        live2d_params = self.live2d_mapper.map_to_live2d(vector, action)
        
        return {
            "emotion_vector": vector.to_dict(),
            "live2d_params": live2d_params.to_dict()
        }
    
    def get_emotion_vector(self, emotion_data: Dict[str, Any]) -> Dict[str, float]:
        vector = self.vector_generator.generate(emotion_data)
        return vector.to_dict()
    
    def get_live2d_params(self, emotion_vector: Dict[str, float], 
                          action: str = None) -> Dict[str, Any]:
        vector = EmotionVector(
            valence=emotion_vector.get("valence", 0.5),
            arousal=emotion_vector.get("arousal", 0.5),
            dominance=emotion_vector.get("dominance", 0.5),
            emotion_label=emotion_vector.get("emotion_label", "neutral"),
            intensity=emotion_vector.get("intensity", 0.5)
        )
        params = self.live2d_mapper.map_to_live2d(vector, action)
        return params.to_dict()
