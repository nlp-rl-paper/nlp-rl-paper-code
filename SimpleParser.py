from vizdoom.vizdoom import GameVariable,GameState
from Parser import Parser


class SimpleParser(Parser):
    def calc_state_string(self,cur_game_state,game_variable):
        state_str = ""
        cur_game_var = game_variable
        #cur_game_state = GameState()
        #cur_game_var = GameVariable()

        state_str += "Health is: " + str(cur_game_var.get_game_variable(GameVariable.HEALTH)) + ", "
        state_str += "Ammo is: " + str(cur_game_var.get_game_variable(GameVariable.AMMO0)) + ", "
        state_str += "hits taken: " + str(cur_game_var.get_game_variable(GameVariable.HITS_TAKEN)) + ", "
        #state_str += "Player position X: " + str(cur_game_state.game_variables[0]) + "Y: " + str(cur_game_state.game_variables[1]) + "Z :" + str(cur_game_state.game_variables[2])
        #parsing enemies on screen

        for l in cur_game_state.labels:
            if l.object_name in self.enemies:
                #state_str += " enemy position X: " + str(l.object_position_x) + " enemy position Y:" +  str(l.object_position_y) +  "enemy position Z:" + str(l.object_position_z)
                state_str += " there is an enemy in " + self.calc_patch(l.object_angle) + ", "
        return state_str

    def calc_patch(self,label_object_angle):
        if label_object_angle < 150:
            return "left patch"
        if label_object_angle > 210:
            return "right patch"
        return "middle patch"