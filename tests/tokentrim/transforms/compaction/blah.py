from agents import Agent, Runner

from dotenv import load_dotenv

from tokentrim import Tokentrim
from tokentrim.transforms import CompactConversation

load_dotenv(override=True)

tt = Tokentrim(tokenizer="gpt-4o-mini")

conversation = [
    {
        "role": "user",
        "content": (
            "I run a small neighborhood bakery and I need help improving our weekly prep workflow. "
            "We bake sourdough, croissants, muffins, cookies, and focaccia."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Happy to help. Tell me how many staff you have, which items cause the most stress, "
            "and whether the bottleneck is mixing, proofing, baking, or packaging."
        ),
    },
    {
        "role": "user",
        "content": (
            "We have four people. Croissants are the worst because lamination takes forever, "
            "proofing gets delayed, and the oven is already busy with sourdough in the morning."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "That sounds like a sequencing problem. Do you currently prep dough the night before, "
            "and are the croissant and sourdough teams sharing the same bench and retarder?"
        ),
    },
    {
        "role": "user",
        "content": (
            "Yes. We bulk-ferment sourdough overnight, but croissant dough sometimes gets mixed too late. "
            "We also share one sheeter, one retarder, and one main oven."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Understood. I would look at separating dough production from shaping and using earlier "
            "lamination windows. Do muffins and cookies compete for the same early-morning labor?"
        ),
    },
    {
        "role": "user",
        "content": (
            "Yes, especially on Fridays. One person portions cookie dough, one scales sourdough, "
            "one laminates, and the last person keeps getting pulled into packaging and front-of-house questions."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "That context helps. We should protect one person from interruptions, move some packaging later, "
            "and create a tighter overnight and first-shift schedule."
        ),
    },
    {
        "role": "user",
        "content": (
            "Please give me a concrete morning plan from 4:00 AM to 9:00 AM that reduces changeovers, "
            "keeps croissants on time, and does not burn out the team."
        ),
    },
]

transform = CompactConversation(model="gpt-5", keep_last=1)
token_budget = 250

preview = tt.compose(transform).apply(
    conversation,
    token_budget=token_budget,
)

print("Preview output tokens:", preview.trace.output_tokens)
print("Preview steps:", [step.step_name for step in preview.trace.steps])
print("Preview context:")
for index, message in enumerate(preview.context, start=1):
    print(f"{index}. {message['role']}: {message['content']}")

run_config = tt.compose(transform).to_openai_agents(
    token_budget=token_budget,
)

try:
    result = Runner.run_sync(
        Agent(name="Assistant", instructions="Answer concisely."),
        conversation,
        run_config=run_config,
    )
    print(result)
except Exception as exc:
    print(f"Runner.run_sync failed after the compaction preview: {exc}")
