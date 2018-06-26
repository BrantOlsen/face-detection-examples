import datetime, os, json, logging
from FaceDetector import FaceObject
import cv2


"""
Represents a collection of faces seen over a period of time that are identified as
the same person.
"""
class FaceIdentifyObject:
    def __init__(self):
        self.faces = []
        self.start_time = datetime.datetime.now()
        self.confirmed_face_id = None
        self.face_movement_threshold = .1
        self.face_count_threshold = 2
        self.face_count_threshold_for_unknown = 7
        self.timeout_threshold = 2
        self.debug_directory = None


    """
    Write all image files for this identifier to a folder.
    """
    def write_debug_files(self, dir: str) -> None:
        debug_directory = os.path.join(dir, self.start_time.strftime("%Y-%m-%d-%H-%M-%S"))

        if not os.path.exists(debug_directory):
            os.makedirs(debug_directory)

        json_obj = {
            'face_id': self.confirmed_face_id,
            'face_count': len(self.faces)
        }

        for face in self.faces:
            file_name = f"face-id-{face.face_id}-time-{face.time.strftime('%H-%M-%S-%f')}"
            path = os.path.join(debug_directory, file_name + '.jpg')
            path_full = os.path.join(debug_directory, file_name + '-full.jpg')
            cv2.imwrite(path, face.image)
            cv2.imwrite(path_full, face.orig_image)
            json_obj[f"face-id-{face.face_id}-time-{face.time.strftime('%H-%M-%S-%f')}"] = {
                'face_id': face.face_id,
                'confidence': face.confidence,
                'all_classifications': face.all_classifications
            }

        json_path = os.path.join(debug_directory, 'data.json')
        json.dump(json_obj, open(json_path, 'w'), indent=4)


    """
    Add a face to the list of faces and if we have not confirmed the identity attempt to
    confirm it.
    """
    def add_face(self, face: FaceObject):
        self.faces.append(face)

        if not self.is_confirmed() and len(self.faces) > self.face_count_threshold:
            counts = {}
            for face in self.faces:
                counts[face.face_id] = (counts[face.face_id] if face.face_id in counts else 0) + 1

            max_count_id = 0
            unknown_id = -1
            for count_id in counts:
                if count_id != unknown_id and (max_count_id == 0 or counts[count_id] > counts[max_count_id]):
                    max_count_id = count_id

            if max_count_id in counts and counts[max_count_id] >= self.face_count_threshold:
                self.confirmed_face_id = max_count_id
            elif unknown_id in counts and counts[unknown_id] > self.face_count_threshold_for_unknown:
                self.confirmed_face_id = unknown_id
            

    """
    Determine if the given gace object is part of this identity object.
    """
    def should_include(self, face: FaceObject) -> bool:
        if self.confirmed_face_id is not None and self.confirmed_face_id == face.face_id:
            return True
        
        last_face = self.faces[len(self.faces) - 1]
        dx = min(last_face.x + last_face.width, face.x + face.width) - max(last_face.x, face.x)
        dy = min(last_face.y + last_face.height, face.y + face.height) - max(last_face.y, face.y)
        if dx >= 0 and dy >= 0:
            return True
            #return dx * dy > face.width * face.height * self.face_movement_threshold

        return False 


    """
    Return True if this identity represents a confirmed face id.
    """
    def is_confirmed(self) -> bool:
        return self.confirmed_face_id is not None


    """
    Get the faceId that this identity represents.
    """
    def get_face_id(self) -> int:
        return self.confirmed_face_id


    """
    Get the average condifidence for all face images.
    """
    def get_confidence(self) -> float:
        if not self.is_confirmed():
            return 0

        sum = 0
        count = 0
        for face in self.faces:
            if face.face_id == self.get_face_id() and face.confidence != 100:
                sum = sum + face.confidence
                count = count + 1

        return sum / count


    """
    Check if this identify object has any relevant faces left. If not, then
    return False.
    """
    def is_expired(self) -> bool:
        cut_off = datetime.datetime.now() - datetime.timedelta(seconds=self.timeout_threshold)
        for face in self.faces:
            if face.time > cut_off:
                return False
        return True