import itertools
from dataclasses import dataclass, field
from typing import Any, Optional

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from src.dtypes import Reward
from src.env import Overcookable, WrappedOvercookedEnv
from src.misc import iter_factory


@dataclass(slots=True, frozen=True)
class OvercookedGame:
    rewards: list[Reward]
    overcooked_states: list[OvercookedState]
    hud_datas: list[dict[str, Any]] = field(default_factory=list)
    env: Optional[Overcookable] = None

    @property
    def base_mdp(self) -> Optional[OvercookedGridworld]:
        if isinstance(self.env, WrappedOvercookedEnv):
            return self.env.base_mdp
        else:
            return None

    def soups_made(self) -> int:
        return sum(reward // 20 for reward in self.rewards)

    def total_reward(self) -> int:
        return sum(self.rewards)

    def visualise(
        self,
        *,
        fps: int = 10,
        wait_after_last_frame: bool = True
    ):
        import pygame

        pygame.init()
        pygame.display.init()

        update_interval = 1000 // fps
        update_event = pygame.USEREVENT + 1

        base_mdp = self.base_mdp
        assert base_mdp is not None

        visualizer = StateVisualizer()
        hud_datas = self.hud_datas or itertools.repeat(None)
        next_rendered_state = iter_factory(
            visualizer.render_state(
                state,
                grid=base_mdp.terrain_mtx,
                hud_data=hud_data
            )
            for state, hud_data in zip(self.overcooked_states, hud_datas)
        )

        rendered_state = next_rendered_state()
        screen = pygame.display.set_mode(
            rendered_state.get_size(), flags=pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.event.set_allowed([pygame.QUIT, update_event])
        screen.blit(rendered_state, (0, 0))
        pygame.display.flip()

        pygame.time.set_timer(update_event, update_interval)
        pygame.display.set_caption("Overcooked-AI")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                elif event.type == update_event:
                    if (rendered_state := next_rendered_state()) is not None:
                        screen.blit(rendered_state, (0, 0))
                        pygame.display.flip()
                    elif not wait_after_last_frame:
                        running = False


@dataclass
class RolloutResults:
    games: list[OvercookedGame]

    def average_reward(self) -> int | float:
        average = 0
        for overcooked_game in self.games:
            average += overcooked_game.total_reward()

        if average == 0:
            return average

        average /= len(self.games)
        return average
