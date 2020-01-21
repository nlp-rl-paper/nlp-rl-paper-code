from vizdoom.vizdoom import GameVariable
from collections import deque


class Parser:
    def __init__(self,hist_size):
        self.state_len = hist_size
        self.history = deque([],hist_size)
        self.enemies_melee = ["Zombieman",
                              "MarineChainsaw",
                              "Cyberdemon",
                              "Demon",
                              "Imp",
                              "BaronBall",
                              "Demon",
                              "HellKnight",
                              "MarineChainsawVzd"
        ]


        self.enemies_ranged = ["DoomImp",
                               "ShotgunGuy",
                               "ChaingunGuy",
                               "Cacodemon"
        ]


        self.ammo_packages = ["ClipBox",
                              "RocketBox",
                              "CellPack",
                              "ShellBox",
                              "Clip"
        ]

        self.weapons = ["RocketLauncher",
                        "Chainsaw",
                        "Chaingun",
                        "SuperShotgun",
                        "Shotgun"
        ]

        self.armors = [
            "RedArmor",
            "GreenArmor",
            "BlueArmor"
        ]

        self.obstacles = [
            "Barrel",
            "BurningBarrel",
            "GrayTree",
            "FloatingSkullRock",
            "EvilEye",
            "ShortGreenPillar",
            "ShortRedPillar",
            "Stalagtite",
            "TallGreenPillar",
            "TallRedPillar",
            "TallTechnoPillar"
        ]

        self.armor_red = [
            "RedArmor"
        ]

        self.armor_green = [
            "GreenArmor"
        ]

        self.armor_blue = [
            "BlueArmor"
        ]

        self.medkits = [
            "Stimpack",
            "Medikit",
            "HealthBonus"
        ]

        self.medkits_stimpack = [
            "Stimpack"
        ]

        self.medkits_medkits = [
            "Medikit"
        ]

        self.obstacles_monsters = [
            "FloatingSkullRock",
            "EvilEye"
        ]

        self.obstacles_barrels = [
            "Barrel",
            "BurningBarrel"
        ]

        self.obstacles_trees = [
            "Stalagtite",
            "GrayTree"
        ]

        self.obstacles_green_pillars = [
            "TallGreenPillar",
            "ShortGreenPillar"
        ]

        self.obstacles_red_pillars = [
            "ShortRedPillar",
            "TallRedPillar"
        ]


        self.powerups = [
            "BlurSphere",
            "InvulnerabilitySphere"
        ]



    def parse_state(self,game_state,game_variable):
        parsed_state = self.calc_state_string(game_state,game_variable)
        self.history.appendleft(parsed_state)
        return self.history

