from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from typesafe_sdk import Noul, Score, Choice, TypeSafeClient

# This is the example directly from the Foundry UI 
# with some slight changes to the "questions" argument of the system_one call.

endpoint = "https://<resource>.services.ai.azure.com/providers/microsoft"
deployment_name = "Decision-1"
token_scope = "https://ai.azure.com/.default"


def main():

    with DefaultAzureCredential() as credential:
        token_provider = get_bearer_token_provider(credential, token_scope)
        with TypeSafeClient(
            api_key=token_provider(),
            base_url=endpoint,
            model=deployment_name,
            timeout=60,
        ) as client:
            
            response = client.system_one(
                state="The payment integration still fails. Please help today.",
                questions={
                    "urgent": Noul(instructions="Does this message convey urgency?"),
                    "request_type": Choice(
                        instructions="What type of request is being made in this message?",
                        criteria={
                            "returns": "Exchanges, wrong or damaged items",
                            "shipping": "Delivery status, delays, lost packages",
                            "billing": "Charges, invoices, payment problems",
                        },
                    ),
                    "satisfaction": Score(
                        instructions="On a scale of 1 to 5, how satisfied is the sender with the current situation?",
                        criteria=[
                            "Cosmetic; no impact to functionality",
                            "Broken or degraded feature, but workaround exists",
                            "Blocking issue; no workaround exists"
                        ]
                    )
                },
            )

            print(response)

    print(f"\nProbability the message conveys urgency: {response.nouls['urgent'].noul:.1%}")
    print(f"Request type: {response.choices['request_type'].choice}")
    print(f"Satisfaction score: {response.scores['satisfaction'].score}")


if __name__ == "__main__":
    main()