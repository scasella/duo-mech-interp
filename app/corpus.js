// A small, varied corpus written for this project (no third-party text). The explorer runs it through the model once
// on the device to measure each neuron's activation range (the scale that "fires" scores use) and to keep the
// contexts where each neuron fires hardest (the Examples tab). Each passage is its own sequence.

export const CORPUS = [
  `The river had flooded twice that spring, and by May the lower fields were still soft enough to swallow a boot. Old Marta walked the levee every morning anyway, counting the herons and muttering about the price of seed. "Nobody plants rice on a promise," she told her grandson, "but nobody plants anything without one."`,
  `def parse_config(path):
    """Read a key=value file, ignoring comments and blank lines."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    return out`,
  `To make the dough, whisk 500 g of flour with 10 g of salt, then add 350 ml of lukewarm water and 7 g of dried yeast. Knead for ten minutes until the dough is smooth and springs back when pressed. Cover the bowl with a damp towel and leave it somewhere warm for about an hour, or until it has doubled in size.`,
  `Dear Mr. Okafor,
Thank you for your letter of 14 March regarding the delayed shipment. I am sorry for the inconvenience this has caused. Our warehouse in Rotterdam was closed for three days because of a strike, and we were unable to notify customers in time. We have refunded the delivery charge and will send the remaining items by express courier at no cost to you.
Kind regards,
Helena Brandt, Customer Relations`,
  `Theorem. If n is an odd integer, then n squared is odd. Proof. Write n = 2k + 1 for some integer k. Then n^2 = 4k^2 + 4k + 1 = 2(2k^2 + 2k) + 1, which is one more than an even number, so n^2 is odd. ∎ The converse also holds: if n^2 is odd, n cannot be even, since the square of an even number is even.`,
  `"You're late," said the captain, without looking up from the chart.
"The tide was against us."
"The tide is always against someone. That is what tides are for." She tapped a finger on the reef marked in red. "We go through here at dawn, or we don't go through at all."`,
  `In 1848, revolutions broke out across Europe in quick succession. Paris rose in February, Vienna and Berlin in March, and within weeks dozens of smaller states had new constitutions or new ministers. Most of these gains were reversed within two years, yet historians still argue about what the year changed: some point to the end of serfdom in the Austrian lands, others to the political education of a generation.`,
  `The match was level at 1–1 going into stoppage time when the substitute, who had been on the pitch for barely four minutes, curled a free kick over the wall and into the top corner. The away fans fell silent. The home bench emptied onto the touchline, and the referee had to wave the coaching staff back before the restart.`,
  `我们周末去了山里的一个小村庄。早上雾很大，几乎看不清对面的山。中午太阳出来以后，村民们把晒好的辣椒和玉米挂在屋檐下，整个村子都变成了红色和金色。晚上我们在一家小饭馆吃了腊肉和野菜，老板娘一直劝我们多喝一碗汤。`,
  `Photosynthesis takes place in two stages. In the light-dependent reactions, chlorophyll absorbs light and the energy is used to split water, releasing oxygen and producing ATP and NADPH. In the Calvin cycle, these molecules drive the fixation of carbon dioxide into three-carbon sugars, which the plant then uses to build glucose, starch and cellulose.`,
  `I keep telling myself it's fine, that everyone feels like this after a move. But it's been six weeks and I still don't know anyone here. Yesterday I said "thank you" to the bus driver and realized it was the first thing I'd said out loud all day. I miss my sister. I miss being known.`,
  `SELECT customer_id, COUNT(*) AS orders, SUM(total) AS revenue
FROM orders
WHERE created_at >= DATE '2025-01-01'
GROUP BY customer_id
HAVING COUNT(*) > 3
ORDER BY revenue DESC
LIMIT 20;`,
  `Terms of Service, Section 7. Limitation of Liability. To the maximum extent permitted by law, the Company shall not be liable for any indirect, incidental, special, consequential or punitive damages, or any loss of profits or revenues, whether incurred directly or indirectly, arising from your use of the Service.`,
  `Q: My tomato plants have yellow leaves at the bottom. Should I worry?
A: A few yellow leaves near the soil are normal as the plant matures. If the yellowing spreads upward or the leaves develop dark spots, it may be early blight. Remove affected leaves, water at the base rather than over the foliage, and mulch to stop soil splashing onto the plant.`,
  `The spacecraft entered orbit at 03:12 UTC after a burn lasting six minutes and forty seconds. Telemetry shows the main engine performed within 0.2 percent of the planned thrust. Over the next three weeks, controllers will lower the orbit in stages until the probe circles the moon at an altitude of about 100 kilometres.`,
  `Honestly? Best concert I've been to in years. The opening band was kind of forgettable but then the lights went down and the whole crowd just screamed. They played the old stuff first, which nobody expected, and when the drummer broke a stick in the middle of the solo he just kept going with one hand lol`,
  `const debounce = (fn, ms = 200) => {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
};
window.addEventListener("resize", debounce(() => layout(window.innerWidth)));`,
  `Le marché ouvre à sept heures, mais les meilleurs fromages sont partis avant huit. Mon voisin, qui vend des pommes depuis quarante ans, dit que les touristes arrivent trop tard et les habitants trop tôt. « Le seul bon moment, » m'a-t-il expliqué, « c'est quand j'ai fini mon café. »`,
  `Patients presenting with chest pain should have an ECG within ten minutes of arrival. Elevated troponin on two measurements taken three hours apart supports a diagnosis of myocardial infarction, but a single normal result does not exclude it. Aspirin should be given unless there is a clear contraindication such as active bleeding or a known allergy.`,
  `The quarterly report showed revenue up 12 percent year over year, driven mainly by subscriptions, while hardware sales fell for the third consecutive quarter. Operating margin narrowed to 18 percent because of higher spending on data centers. Management raised its full-year guidance but warned that currency movements could reduce reported growth.`,
  `Once upon a time a fox found a crow sitting on a branch with a piece of cheese in its beak. "What a beautiful voice you must have," said the fox. The crow, flattered, opened its beak to sing, and the cheese fell straight into the fox's mouth. Never trust a compliment that wants something from you.`,
  `Error: Cannot read properties of undefined (reading 'map')
    at renderList (app.js:214:23)
    at HTMLButtonElement.<anonymous> (app.js:388:5)
The list is undefined because the request failed silently. Check the network tab: the server returned 401, so the token has probably expired. Log out and back in, then try again.`,
  `Wait, so you're telling me the meeting was moved to Thursday and nobody told me? I literally drove in on my day off. No, no, it's fine. It's totally fine. I'll just sit here in the empty conference room with my coffee and my printed agenda like an idiot.`,
  `Glaciers advance and retreat over thousands of years, carving valleys into a characteristic U shape and leaving ridges of rock and gravel called moraines where they stop. When the ice melts, the hollows it scooped out often fill with water; many of the lakes in Finland, Canada and the northern United States formed this way at the end of the last ice age.`,
  `Hey! Just landed 🛬 Taxi line is huge so might be 40 min. Can you put the kettle on? Also don't let the dog eat my birthday cake again, I'm serious this time 😂🎂`,
  `Article 3. Every person has the right to rest and leisure, including a reasonable limitation of working hours. No one shall be required to perform labour for more than forty-eight hours in any week, except in emergencies defined by law and with compensation at not less than one and a half times the ordinary rate.`,
  `He had always hated the sound of the refrigerator at night, that low electric hum that seemed to grow louder the longer he listened. Now, alone in the apartment for the first time since the funeral, he found he was grateful for it. It meant something in the house was still working.`,
  `The chess opening known as the Queen's Gambit begins 1. d4 d5 2. c4. White offers a pawn to draw Black's central pawn away from the center. If Black accepts with 2...dxc4, White usually regains the pawn within a few moves; if Black declines with 2...e6, the game tends toward slower positional play.`,
];
