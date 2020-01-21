from vizdoom.vizdoom import GameVariable,GameState
import numpy as np
from Parser import Parser

HEALTH_LOW_RATE = 10
AMMO_LOW_RATE = 5
ENEMIES_HIGH_RATE = 2
CENTER_ANGLE_RANGE = 20
RIGHT_ANGLE_RANGE = 60
SCREEN_WIDTH_CENTER = 320
SCREEN_HEIGHT_CENTER = 240
SCREEN_WIDTH = 2*SCREEN_WIDTH_CENTER
BLOOD_HIT_PLAYER_THRESHOLD = 100
PATCHES_COUNT = 3
CLOSE_TO_PLAYER_THRESHOLD = 200
FAR_FROM_PLAYER_THRESHOLD = 300
DOOM_IMPBALL_CLOSE = 200
DOOM_IMPBALL_FAR = 350

class RoyParser(Parser):
    def __init__(self,hist_size,reverse_green=False):
        super(RoyParser,self).__init__(hist_size)
        self.init = 1
        self.player_angle = 0
        self.player_xy = 0,0
        self.total_hits_taken = 0
        self.total_hit_shot = 0
        self.kills = 0
        self.total_outgoing_damage = 0
        self.ammo = 0
        self.health = 0
        self.weapon_equiped = 0
        self.reverse_green = reverse_green


    def calc_state_string(self,cur_game_state : GameState ,game_variable : GameVariable):
        state_str = ""
        if self.init == 1:
            self.ammo = game_variable.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
            self.health = game_variable.get_game_variable(GameVariable.HEALTH)
            self.init = 0
            self.weapon_equiped = 2

        health = game_variable.get_game_variable(GameVariable.HEALTH)
        ammo = game_variable.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        hits_taken = game_variable.get_game_variable(GameVariable.HITS_TAKEN)
        hit_shot = game_variable.get_game_variable(GameVariable.HITCOUNT)
        kills = game_variable.get_game_variable(GameVariable.KILLCOUNT)
        weapon_equiped = game_variable.get_game_variable(GameVariable.SELECTED_WEAPON)

        self.player_xy = np.array([game_variable.get_game_variable(GameVariable.POSITION_X),game_variable.get_game_variable(GameVariable.POSITION_Y)])  # doomplayer

        state_str += "This is a first person view, "
        state_str += self.calc_health_string(health)
        if weapon_equiped != self.weapon_equiped:
            state_str += "player has equiped a new weapon! "
            self.weapon_equiped = weapon_equiped

        if health > self.health:
            state_str += "the player healed itself using a healthcare package"

        self.health = health

        if ammo <= 0:
            state_str += "the player is out of ammo"
        else:
            state_str += "the character has "+ str(int(ammo)) + " rounds in the magazine. "
        #state_str += "you were hit " + hits_taken + "times in total. "

        #parsing enemies on screen
        enemy_hit_flag = 0
        player_hit_flag = 0
        player_dodge_flag = 0
        num_enemies = 0
        weapon_equip_flag = 0



        #enemy counter dictionaries: 0 - close, 1 - mid , 2 - far
        fireball_patch_counter = [{-1: 0, 0: 0, 1: 0},
                                      {-1: 0, 0: 0, 1: 0},
                                      {-1: 0, 0: 0, 1: 0}]


        ranged_enemy_patch_counter = [{-1: 0, 0: 0, 1: 0},
                               {-1: 0, 0: 0, 1: 0},
                               {-1: 0, 0: 0, 1: 0}]

        melee_enemy_patch_counter = [{-1: 0, 0: 0, 1: 0},
                               {-1: 0, 0: 0, 1: 0},
                               {-1: 0, 0: 0, 1: 0}]

        medkit_patch_counter = [{-1: 0, 0: 0, 1: 0},
                               {-1: 0, 0: 0, 1: 0},
                               {-1: 0, 0: 0, 1: 0}]


        armor_red_patch_counter = [{-1: 0, 0: 0, 1: 0},
                               {-1: 0, 0: 0, 1: 0},
                               {-1: 0, 0: 0, 1: 0}]

        armor_blue_patch_counter = [{-1: 0, 0: 0, 1: 0},
                               {-1: 0, 0: 0, 1: 0},
                               {-1: 0, 0: 0, 1: 0}]

        armor_green_patch_counter = [{-1: 0, 0: 0, 1: 0},
                                    {-1: 0, 0: 0, 1: 0},
                                    {-1: 0, 0: 0, 1: 0}]

        weapons_patch_counter = [{-1: 0, 0: 0, 1: 0},
                               {-1: 0, 0: 0, 1: 0},
                               {-1: 0, 0: 0, 1: 0}]

        ammo_patch_counter = [{-1: 0, 0: 0, 1: 0},
                               {-1: 0, 0: 0, 1: 0},
                               {-1: 0, 0: 0, 1: 0}]

        medkit_medkit_patch_counter = [{-1: 0, 0: 0, 1: 0},
                              {-1: 0, 0: 0, 1: 0},
                              {-1: 0, 0: 0, 1: 0}]

        medkit_stimpack_patch_counter = [{-1: 0, 0: 0, 1: 0},
                                       {-1: 0, 0: 0, 1: 0},
                                       {-1: 0, 0: 0, 1: 0}]


        obstacles_monsters_patch_counter = [{-1: 0, 0: 0, 1: 0},
                                           {-1: 0, 0: 0, 1: 0},
                                           {-1: 0, 0: 0, 1: 0}]

        obstacles_barrels_patch_counter = [{-1: 0, 0: 0, 1: 0},
                                   {-1: 0, 0: 0, 1: 0},
                                   {-1: 0, 0: 0, 1: 0}]

        obstacles_trees_patch_counter = [{-1: 0, 0: 0, 1: 0},
                                   {-1: 0, 0: 0, 1: 0},
                                   {-1: 0, 0: 0, 1: 0}]

        obstacles_green_pillars_counter = [{-1: 0, 0: 0, 1: 0},
                                   {-1: 0, 0: 0, 1: 0},
                                   {-1: 0, 0: 0, 1: 0}]

        obstacles_red_pillars_counter = [{-1: 0, 0: 0, 1: 0},
                                   {-1: 0, 0: 0, 1: 0},
                                   {-1: 0, 0: 0, 1: 0}]

        invul_patch_counter = [{-1: 0, 0: 0, 1: 0},
                              {-1: 0, 0: 0, 1: 0},
                              {-1: 0, 0: 0, 1: 0}]

        stealth_patch_counter = [{-1: 0, 0: 0, 1: 0},
                               {-1: 0, 0: 0, 1: 0},
                               {-1: 0, 0: 0, 1: 0}]



        patch_str_dict = {
            #-2: "outer left ",
            -1: "left ",
            0: "front ",
            1: "right ",
            #2: "outer right "
        }

        patch_dist_str_dict = {
            0 : "near ",
            1 : "",
            2 : "far "
        }



        if hits_taken > self.total_hits_taken:
            state_str += "you were hit! "
            self.total_hits_taken = hits_taken
            player_hit_flag = 1

        for label_index,l in enumerate(cur_game_state.labels):
            #we ignore the player
            if l.object_name is not "DoomPlayer":
                '''
                for each label:
                    calculate screen patch
                    calculate distance
                    update counter
                '''

                if l.object_name in self.enemies_melee:
                    is_close = self.calc_close_to_player_string(cur_game_state,label_index)
                    patch_id, path_str = self.calc_patch(l.x)
                    melee_enemy_patch_counter[is_close][patch_id] += 1
                    num_enemies += 1

                if l.object_name in self.enemies_ranged:
                    is_close = self.calc_close_to_player_string(cur_game_state,label_index)
                    patch_id, path_str = self.calc_patch(l.x)
                    ranged_enemy_patch_counter[is_close][patch_id] += 1
                    num_enemies += 1


                elif l.object_name == "BulletPuff":
                    puff_xy = np.array([l.object_position_x, l.object_position_y])
                    if np.linalg.norm(puff_xy - self.player_xy) < BLOOD_HIT_PLAYER_THRESHOLD:  # the blood is close to the player, so the player is hit
                        player_dodge_flag = 1
                        state_str += "the enemy fire missed! "
                    else:
                        state_str += "the player has not aimed well. "

                elif l.object_name == "DoomImpBall":
                    fireball_distance = self.calc_distance_string(cur_game_state,label_index)
                    patch_id, path_str = self.calc_patch(l.x)
                    state_str += "a monster shot an energy bomb " + path_str + ", "
                    if fireball_distance < DOOM_IMPBALL_CLOSE:
                        state_str += "and it is really close, "
                    elif fireball_distance > DOOM_IMPBALL_FAR:
                        state_str += "and it is far away "


                elif l.object_name in self.ammo_packages:
                    patch_id, path_str = self.calc_patch(l.x)
                    is_close = self.calc_close_to_player_string(cur_game_state,label_index)
                    ammo_patch_counter[is_close][patch_id] += 1

                elif l.object_name in self.armor_red:
                    patch_id, path_str = self.calc_patch(l.x)
                    is_close = self.calc_close_to_player_string(cur_game_state,label_index)
                    armor_red_patch_counter[is_close][patch_id] += 1

                elif l.object_name in self.armor_green:
                    patch_id, path_str = self.calc_patch(l.x)
                    is_close = self.calc_close_to_player_string(cur_game_state,label_index)
                    armor_green_patch_counter[is_close][patch_id] += 1

                elif l.object_name in self.armor_blue:
                    patch_id, path_str = self.calc_patch(l.x)
                    is_close = self.calc_close_to_player_string(cur_game_state,label_index)
                    armor_blue_patch_counter[is_close][patch_id] += 1

                elif l.object_name in self.weapons:
                    patch_id, path_str = self.calc_patch(l.x)
                    is_close = self.calc_close_to_player_string(cur_game_state,label_index)
                    weapons_patch_counter[is_close][patch_id] += 1

                elif l.object_name in self.medkits_medkits:
                    patch_id, path_str = self.calc_patch(l.x)
                    is_close = self.calc_close_to_player_string(cur_game_state,label_index)
                    medkit_medkit_patch_counter[is_close][patch_id] += 1

                elif l.object_name in self.medkits_stimpack:
                    patch_id, path_str = self.calc_patch(l.x)
                    is_close = self.calc_close_to_player_string(cur_game_state,label_index)
                    medkit_stimpack_patch_counter[is_close][patch_id] += 1

                elif l.object_name in self.obstacles_monsters:
                    patch_id, path_str = self.calc_patch(l.x)
                    is_close = self.calc_close_to_player_string(cur_game_state,label_index)
                    obstacles_monsters_patch_counter[is_close][patch_id] += 1

                elif l.object_name in self.obstacles_barrels:
                    patch_id, path_str = self.calc_patch(l.x)
                    is_close = self.calc_close_to_player_string(cur_game_state,label_index)
                    obstacles_barrels_patch_counter[is_close][patch_id] += 1

                elif l.object_name in self.obstacles_trees:
                    patch_id, path_str = self.calc_patch(l.x)
                    is_close = self.calc_close_to_player_string(cur_game_state,label_index)
                    obstacles_trees_patch_counter[is_close][patch_id] += 1

                elif l.object_name in self.obstacles_red_pillars:
                    patch_id, path_str = self.calc_patch(l.x)
                    is_close = self.calc_close_to_player_string(cur_game_state,label_index)
                    obstacles_red_pillars_counter[is_close][patch_id] += 1

                elif l.object_name in self.obstacles_green_pillars:
                    patch_id, path_str = self.calc_patch(l.x)
                    is_close = self.calc_close_to_player_string(cur_game_state,label_index)
                    obstacles_green_pillars_counter[is_close][patch_id] += 1

                elif l.object_name == "BlurSphere":
                    patch_id, path_str = self.calc_patch(l.x)
                    is_close = self.calc_close_to_player_string(cur_game_state,label_index)
                    stealth_patch_counter[is_close][patch_id] += 1

                elif l.object_name == "InvulnerabilitySphere":
                    patch_id, path_str = self.calc_patch(l.x)
                    is_close = self.calc_close_to_player_string(cur_game_state,label_index)
                    invul_patch_counter[is_close][patch_id] += 1


        '''
        generate the strings after processing the labels
        '''

        state_str += self.calc_player_shot_string(ammo,hit_shot,kills)

        for dist in range(0,3):
            for i in range(-1,2):
                patch_dist_counter = 0
                if melee_enemy_patch_counter[dist][i] > 0:
                    patch_dist_counter += 1
                    if melee_enemy_patch_counter[dist][i] == 1:
                        state_str += " there is a bad guy, "
                    else:
                        state_str += " there are " + str(melee_enemy_patch_counter[dist][i]) + " bad guys,"
                if ranged_enemy_patch_counter[dist][i] > 0:
                    patch_dist_counter += 1
                    if ranged_enemy_patch_counter[dist][i] == 1:
                        state_str += " there is a fireball shooting bad guy,"
                    else:
                        state_str += " there are " + str(ranged_enemy_patch_counter[dist][i]) + " fireball shooting bad guys,"
                if medkit_medkit_patch_counter[dist][i] > 0:
                    patch_dist_counter += 1
                    if medkit_medkit_patch_counter[dist][i] == 1:
                        state_str += " there is a health package,"
                    else:
                        state_str += " there are " + str(medkit_medkit_patch_counter[dist][i]) + " first health packages,"
                if medkit_stimpack_patch_counter[dist][i] > 0:
                    patch_dist_counter += 1
                    if medkit_stimpack_patch_counter[dist][i] == 1:
                        state_str += " there is a small health potion,"
                    else:
                        state_str += " there are " + str(medkit_stimpack_patch_counter[dist][i]) + "small health potions,"

                if ammo_patch_counter[dist][i] > 0:
                    patch_dist_counter += 1
                    if ammo_patch_counter[dist][i] == 1:
                        state_str += " there is a new magazine,"
                    else:
                        state_str += " there are " + str(ammo_patch_counter[dist][i]) + " new magazines,"

                if armor_green_patch_counter[dist][i] > 0:
                    patch_dist_counter += 1
                    if armor_green_patch_counter[dist][i] == 1:
                        state_str += " there is a green shield,"
                    else:
                        state_str += " there are " + str(armor_green_patch_counter[dist][i]) + " green shields,"

                if armor_red_patch_counter[dist][i] > 0:
                    patch_dist_counter += 1
                    if armor_red_patch_counter[dist][i] == 1:
                        state_str += " there is a red shield,"
                    else:
                        state_str += " there are " + str(armor_red_patch_counter[dist][i]) + " red shields,"

                if armor_blue_patch_counter[dist][i] > 0:
                    patch_dist_counter += 1
                    if armor_blue_patch_counter[dist][i] == 1:
                        state_str += " there is a blue shield,"
                    else:
                        state_str += " there are " + str(armor_blue_patch_counter[dist][i]) + " blue shields,"

                if weapons_patch_counter[dist][i] > 0:
                    patch_dist_counter += 1
                    if weapons_patch_counter[dist][i] == 1:
                        state_str += " there is a new gun,"
                    else:
                        state_str += " there are " + str(weapons_patch_counter[dist][i]) + " new guns,"

                if obstacles_monsters_patch_counter[dist][i] > 0:
                    patch_dist_counter += 1
                    if obstacles_monsters_patch_counter[dist][i] == 1:
                        state_str += " there is a sarcophage "
                    else:
                        state_str += " there are " + str(obstacles_monsters_patch_counter[dist][i]) + " sarcophages,"

                if obstacles_barrels_patch_counter[dist][i] > 0:
                    patch_dist_counter += 1
                    if obstacles_barrels_patch_counter[dist][i] == 1:
                        state_str += " there is an acid cask "
                    else:
                        state_str += " there are " + str(obstacles_barrels_patch_counter[dist][i]) + " acid caskss,"

                if obstacles_trees_patch_counter[dist][i] > 0:
                    patch_dist_counter += 1
                    if obstacles_trees_patch_counter[dist][i] == 1:
                        state_str += " there is a dead bush "
                    else:
                        state_str += " there are " + str(obstacles_trees_patch_counter[dist][i]) + " dead bush,"

                if obstacles_green_pillars_counter[dist][i] > 0:
                    patch_dist_counter += 1
                    if obstacles_green_pillars_counter[dist][i] == 1:
                        state_str += " there is a green statue "
                    else:
                        state_str += " there are " + str(obstacles_green_pillars_counter[dist][i]) + " green statues,"

                if obstacles_red_pillars_counter[dist][i] > 0:
                    patch_dist_counter += 1
                    if obstacles_red_pillars_counter[dist][i] == 1:
                        state_str += " there is a red statue "
                    else:
                        state_str += " there are " + str(obstacles_red_pillars_counter[dist][i]) + " red statues,"

                if stealth_patch_counter[dist][i] > 0:
                    patch_dist_counter += 1
                    if stealth_patch_counter[dist][i] == 1:
                        state_str += " there is a camouflage kit "
                    else:
                        state_str += " there are " + str(stealth_patch_counter[dist][i]) + " camouflage kits"

                if invul_patch_counter[dist][i] > 0:
                    patch_dist_counter += 1
                    if invul_patch_counter[dist][i] == 1:
                        state_str += " there is a god mod sphere "
                    else:
                        state_str += " there are " + str(invul_patch_counter[dist][i]) + " hod mod spheres "

                if patch_dist_counter > 0:
                    patch_dist_counter += 1
                    state_str = state_str[:-1]
                    state_str +=  " to the player's " + patch_dist_str_dict[dist] + patch_str_dict[i] + ", "



        if num_enemies == 0:
            state_str += "the player is alone "
            if player_hit_flag == 1 or player_dodge_flag == 1:
                state_str += ", but is somehow taking damage!"
            else:
                state_str += "."

        #if num_enemies > ENEMIES_HIGH_RATE:
            #state_str += "you have a lot of enemies. "
        return state_str

    def get_xy_of_label(self, cur_game_state,label_id):
        if len(cur_game_state.labels) > 0:
            return np.array([cur_game_state.labels[label_id].object_position_x, cur_game_state.labels[label_id].object_position_y])
        return np.array([0,0])


    def get_screen_xy_of_label(self,cur_game_state, label_id):
        return np.array([cur_game_state.labels[label_id].x, cur_game_state.labels[label_id].y])


    def calc_patch(self, x):
        if x < SCREEN_WIDTH * 4 / 9:
            return -1, "to the left part of screen"
        elif x < SCREEN_WIDTH * 5 / 9:
            return 0, "in center of screen "
        return 1, "to the right part of screen"


    def calc_health_string (self,health):
        if health == 100:
            return "player has maximum hit points. "
        if health > 60:
            return "player has many hit points left. "
        if health > 30:
            return "player has some hit points left! "
        return "player has few hit points! "


    def calc_distance_string(self,cur_game_state,label_index):
        x, y = self.get_xy_of_label(cur_game_state, label_index)
        return np.sqrt((x - self.player_xy[0]) ** 2 + (y - self.player_xy[1]) ** 2)


    def calc_close_to_player_string(self,cur_game_state,label_index):
        x,y = self.get_xy_of_label(cur_game_state,label_index)
        distance = np.sqrt((x - self.player_xy[0])**2 + (y - self.player_xy[1])**2)
        if distance < CLOSE_TO_PLAYER_THRESHOLD:
            return 0
        if distance > FAR_FROM_PLAYER_THRESHOLD:
            return 2
        return 1


    def calc_player_shot_string(self,ammo,hits,kill_count):
        result_str = ""
        if ammo < self.ammo: # a shot was made"
            if hits > self.total_hit_shot: #it hit the target
                result_str += "player shot "
                if kill_count > self.kills: #and it was killed
                    result_str += "and killed a bad guy! "
                else: #you hit but didnt kill
                    result_str += "a target "
            else:
                result_str += "and aimed at the wall! "
        self.ammo = ammo
        self.total_hit_shot = hits
        self.total_hits_taken = kill_count
        return result_str


    def calc_player_dogde(self):
        raise NotImplementedError