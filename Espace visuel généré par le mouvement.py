import cv2
import numpy as np
import pygame
import math
import random
from collections import deque
import mediapipe as mp

# Initialisation Pygame
pygame.init()
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Espace Visuel Généré par le Mouvement - Avec Silhouette")
clock = pygame.time.Clock()

# Initialisation MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Capture vidéo
cap = cv2.VideoCapture(0)

class Particle:
    def __init__(self, x, y, movement_data):
        self.x = x
        self.y = y
        self.vx = movement_data['direction_x'] * 2 + random.uniform(-1, 1)
        self.vy = movement_data['direction_y'] * 2 + random.uniform(-1, 1)
        self.life = 255
        self.color = self.get_color_from_movement(movement_data)
        self.size = movement_data['intensity'] * 5 + 2
        self.trail = deque(maxlen=10)
        
    def get_color_from_movement(self, movement_data):
        r = min(255, int(abs(movement_data['direction_x']) * 255))
        g = min(255, int(abs(movement_data['direction_y']) * 255))
        b = min(255, int(movement_data['intensity'] * 255))
        return (r, g, b)
    
    def update(self):
        self.trail.append((self.x, self.y))
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.98
        self.vy *= 0.98
        self.life -= 3
        self.size *= 0.99
        
    def draw(self, surface):
        if len(self.trail) > 1:
            for i in range(1, len(self.trail)):
                alpha = int(self.life * (i / len(self.trail)))
                pos1 = self.trail[i-1]
                pos2 = self.trail[i]
                color = (*self.color[:3], alpha)
                pygame.draw.line(surface, color, pos1, pos2, max(1, int(self.size/2)))
        
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.size))

class MotionVisualGenerator:
    def __init__(self):
        self.particles = []
        self.previous_landmarks = None
        self.movement_history = deque(maxlen=10)
        self.silhouette_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        self.camera_surface = pygame.Surface((WIDTH, HEIGHT))
        
    def detect_motion_parameters(self, landmarks):
        movement_data = {
            'direction_x': 0,
            'direction_y': 0,
            'speed': 0,
            'intensity': 0
        }
        
        if landmarks and self.previous_landmarks:
            total_movement = 0
            avg_dx, avg_dy = 0, 0
            point_count = 0
            
            for prev_lm, curr_lm in zip(self.previous_landmarks, landmarks):
                if prev_lm.visibility > 0.7 and curr_lm.visibility > 0.7:
                    dx = curr_lm.x - prev_lm.x
                    dy = curr_lm.y - prev_lm.y
                    movement = math.sqrt(dx*dx + dy*dy)
                    
                    avg_dx += dx
                    avg_dy += dy
                    total_movement += movement
                    point_count += 1
            
            if point_count > 0:
                movement_data['direction_x'] = avg_dx / point_count * 10
                movement_data['direction_y'] = avg_dy / point_count * 10
                movement_data['speed'] = total_movement / point_count * 100
                movement_data['intensity'] = min(1.0, total_movement / point_count * 50)
        
        self.previous_landmarks = landmarks
        return movement_data
    
    def draw_silhouette(self, landmarks, movement_data):
        # Effacer la surface de silhouette
        self.silhouette_surface.fill((0, 0, 0, 0))
        
        if landmarks:
            # Couleur de la silhouette basée sur l'intensité du mouvement
            silhouette_color = (
                min(255, int(movement_data['intensity'] * 255)),
                min(255, int(abs(movement_data['direction_x']) * 255)),
                min(255, int(abs(movement_data['direction_y']) * 255)),
                150  # Transparence
            )
            
            # Dessiner les connexions du corps
            connections = [
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
                (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
                (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
                (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
                (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
                (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
            ]
            
            # Dessiner les lignes de connexion
            for connection in connections:
                start_lm = landmarks[connection[0].value]
                end_lm = landmarks[connection[1].value]
                
                if start_lm.visibility > 0.7 and end_lm.visibility > 0.7:
                    start_pos = (int(start_lm.x * WIDTH), int(start_lm.y * HEIGHT))
                    end_pos = (int(end_lm.x * WIDTH), int(end_lm.y * HEIGHT))
                    
                    line_width = max(2, int(movement_data['intensity'] * 8))
                    pygame.draw.line(self.silhouette_surface, silhouette_color, 
                                   start_pos, end_pos, line_width)
            
            # Dessiner les points de jointure
            for landmark in landmarks:
                if landmark.visibility > 0.7:
                    pos = (int(landmark.x * WIDTH), int(landmark.y * HEIGHT))
                    point_size = max(3, int(movement_data['intensity'] * 10))
                    pygame.draw.circle(self.silhouette_surface, silhouette_color, 
                                     pos, point_size)
    
    def update_camera_view(self, frame):
        # Convertir l'image OpenCV en surface Pygame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (WIDTH, HEIGHT))
        frame_rotated = cv2.rotate(frame_resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Appliquer un effet visuel à l'image de la caméra
        frame_effect = self.apply_camera_effect(frame_rotated)
        
        # Convertir en surface Pygame
        camera_surface = pygame.surfarray.make_surface(frame_effect)
        self.camera_surface = pygame.transform.flip(camera_surface, True, False)
    
    def apply_camera_effect(self, frame):
        """Applique un effet visuel à l'image de la caméra"""
        # Effet de contraste augmenté
        frame_float = frame.astype(np.float32)
        frame_contrast = np.clip((frame_float - 128) * 1.5 + 128, 0, 255)
        
        # Légère teinte bleutée
        frame_contrast[:, :, 2] = np.clip(frame_contrast[:, :, 2] * 0.8, 0, 255)
        frame_contrast[:, :, 0] = np.clip(frame_contrast[:, :, 0] * 1.1, 0, 255)
        
        return frame_contrast.astype(np.uint8)
    
    def generate_particles(self, movement_data, landmarks):
        if landmarks:
            key_points = [mp_pose.PoseLandmark.LEFT_SHOULDER, 
                         mp_pose.PoseLandmark.RIGHT_SHOULDER,
                         mp_pose.PoseLandmark.LEFT_HIP,
                         mp_pose.PoseLandmark.RIGHT_HIP,
                         mp_pose.PoseLandmark.NOSE,
                         mp_pose.PoseLandmark.LEFT_WRIST,
                         mp_pose.PoseLandmark.RIGHT_WRIST]
            
            for point in key_points:
                lm = landmarks[point.value]
                if lm.visibility > 0.7:
                    screen_x = int(lm.x * WIDTH)
                    screen_y = int(lm.y * HEIGHT)
                    
                    if random.random() < movement_data['intensity'] * 0.3:
                        for _ in range(int(movement_data['intensity'] * 2)):
                            self.particles.append(Particle(screen_x, screen_y, movement_data))
    
    def update_particles(self):
        for particle in self.particles[:]:
            particle.update()
            if (particle.life <= 0 or particle.x < -100 or particle.x > WIDTH + 100 or 
                particle.y < -100 or particle.y > HEIGHT + 100):
                self.particles.remove(particle)
    
    def draw_visuals(self, movement_data):
        # Fond avec effet de fade
        fade_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        fade_surface.fill((0, 0, 0, 15))
        screen.blit(fade_surface, (0, 0))
        
        # Afficher la vue caméra en arrière-plan (optionnel - décommentez pour voir)
        # screen.blit(self.camera_surface, (0, 0))
        
        # Dessiner les particules
        for particle in self.particles:
            particle.draw(screen)
        
        # Dessiner la silhouette par-dessus
        screen.blit(self.silhouette_surface, (0, 0))
        
        # Affichage des données
        font = pygame.font.Font(None, 36)
        texts = [
            f"Intensité: {movement_data['intensity']:.2f}",
            f"Vitesse: {movement_data['speed']:.2f}",
            f"Particules: {len(self.particles)}",
            "Bougez pour générer des effets visuels!"
        ]
        
        for i, text in enumerate(texts):
            text_surface = font.render(text, True, (255, 255, 255))
            screen.blit(text_surface, (10, 10 + i * 30))
    
    def run(self):
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Mettre à jour la vue caméra
            self.update_camera_view(frame)
            
            # Traitement MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            landmarks = None
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
            
            # Détection des paramètres de mouvement
            movement_data = self.detect_motion_parameters(landmarks)
            
            # Dessiner la silhouette
            self.draw_silhouette(landmarks, movement_data)
            
            # Génération de particules
            self.generate_particles(movement_data, landmarks)
            
            # Mise à jour des particules
            self.update_particles()
            
            # Rendu
            self.draw_visuals(movement_data)
            pygame.display.flip()
            clock.tick(60)
        
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    generator = MotionVisualGenerator()
    generator.run()