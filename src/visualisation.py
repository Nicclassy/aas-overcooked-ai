from collections.abc import Iterable

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from src.misc import iter_factory
from src.types_ import Reward


def plot_rewards(*values: Iterable[Reward]):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    for agent_number, rewards in enumerate(values, start=1):
        plt.plot(rewards, label=f"Agent {agent_number}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO rewards by agent")
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def visualise_game(
    states: Iterable[OvercookedState], base_mdp: OvercookedGridworld, fps: int = 10
):
    import pygame

    pygame.init()
    pygame.display.init()

    update_interval = 1000 // fps
    update_event = pygame.USEREVENT + 1

    visualizer = StateVisualizer()
    next_rendered_state = iter_factory(
        visualizer.render_state(state, grid=base_mdp.terrain_mtx) for state in states
    )

    rendered_state = next_rendered_state()
    screen = pygame.display.set_mode(
        rendered_state.get_size(), flags=pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    screen.blit(rendered_state, (0, 0))
    pygame.display.flip()

    pygame.time.set_timer(update_event, update_interval)
    pygame.display.set_caption("Overcooked-AI")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
            elif event.type == update_event:
                if (rendered_state := next_rendered_state()) is not None:
                    screen.blit(rendered_state, (0, 0))
                    pygame.display.flip()
                else:
                    running = False
