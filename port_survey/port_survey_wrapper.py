import port_survey.boat_dbms as dbms


class PortSurvey:
    def __init__(self, boat_classify_model, hin_model, water_level_model):
        self.boat_classify_model = boat_classify_model
        self.hin_model = hin_model

    def get_inference_data(self, img):
        return {
            "person_count": 10,
            "Water Level": 5,
        }

    def get_json_data(self):
        return {}
