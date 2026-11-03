# Targeted sweep: ordered and cyclic word families nobody has checked in a model's geometry.
#   python tools/shapes/family_sweep.py <model dir or HF id> <out dir> [layers] [--no-steer] [--cache DIR]
# For each family (values in a known order, 8 template sentences, a copy prompt):
#   1. Residual at the value word (last token of the word; multi-token words allowed), per template and layer;
#      template-centred per-value means.
#   2. Order: rank correlation of the top principal axis with the known order (linear), or neighbours kept adjacent
#      in angle on the top plane (cyclic); permutation p-values, also for the best of the top 3 axes / planes.
#      Frequency control: the same correlation after partialling out each word's Zipf frequency (wordfreq).
#   3. Shape: with the copy prompt, the residual at the word is moved (a) to each value's own average ("node check":
#      can the model produce the middle words at all from this prompt?) and (b) along the straight line from the first
#      value's average to the last ("chord"). A straight scale passes through the reproducible middle words on the way;
#      a curved one skips them (as the weekday circle and the "how soon" scale do).
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer, AutoModelForCausalLM

L_ = "linear"; C_ = "cyclic"
def F(order, vals, templates, copy, **kw):
    return dict(order=order, vals=vals.split(), templates=templates, copy=copy, **kw)

WORD_COPY = "She wrote the word {X} on the board, and he also wrote the word"
FAMILIES = {
    # --- hierarchies
    "military rank": F(L_, "private corporal sergeant lieutenant captain major colonel general", [
        "The {X} gave the order to advance.", "His rank was {X} when the war ended.", "The {X} saluted the flag.",
        "Everyone in the unit respected the {X}.", "The {X} inspected the barracks.", "The young {X} wrote a letter home.",
        "A letter arrived for the {X}.", "The {X} led the soldiers across the river."],
        "The soldier's rank was {X}, and his brother's rank was also"),
    "church rank": F(L_, "deacon priest bishop archbishop cardinal pope", [
        "The {X} led the service on Sunday.", "The {X} blessed the congregation.", "She met the {X} after mass.",
        "The {X} visited the parish.", "The letter was signed by the {X}.", "The {X} spoke about forgiveness.",
        "Crowds gathered to see the {X}.", "The {X} wore a long robe."],
        "The service was led by the {X}, and the blessing was also given by the"),
    "nobility": F(L_, "baron viscount earl marquis duke prince king emperor", [
        "The {X} held a feast at the castle.", "The {X} owned vast lands.", "Peasants bowed before the {X}.",
        "The {X} rode into the village.", "The title of {X} passed to his son.", "The {X} signed the treaty.",
        "The {X} lived in a grand palace.", "Servants waited on the {X}."],
        "The castle belonged to the {X}, and the forest also belonged to the"),
    "corporate rank": F(L_, "intern assistant manager director CEO", [
        "The {X} scheduled the meeting.", "The {X} approved the budget.", "The {X} works on the fifth floor.",
        "Everyone waited for the {X} to arrive.", "The {X} sent an email to the team.", "The {X} reviewed the report.",
        "Nobody wanted to tell the {X} the bad news.", "The {X} signed the contract."],
        "The memo came from the {X}, and the reply also came from the"),
    "academic rank": F(L_, "student lecturer professor dean provost president", [
        "The {X} gave a speech at graduation.", "The {X} met with the committee.", "The {X} published a paper.",
        "Everyone stood when the {X} entered.", "The {X} approved the new course.", "The {X} has an office in the library.",
        "The letter from the {X} arrived today.", "The {X} answered questions after the talk."],
        "The award was given by the {X}, and the prize was also given by the"),
    "chess piece": F(L_, "pawn knight bishop rook queen king", [
        "She moved her {X} forward.", "He captured the {X} on the next move.", "White sacrificed the {X} to win the game.",
        "The {X} was trapped on the edge of the board.", "Protect your {X} at all costs.", "Black lost a {X} early in the game.",
        "The {X} controls the center.", "My opponent attacked my {X}."],
        "White moved the {X}, and then Black also moved the"),
    "card rank": F(L_, "ten jack queen king ace", [
        "He played the {X} of hearts.", "She drew the {X} of spades.", "The {X} of clubs was on the table.",
        "I'm holding the {X} of diamonds.", "The dealer turned over the {X} of hearts.", "He was missing the {X} of spades.",
        "The magician found the {X} of clubs.", "She slapped down the {X} of diamonds."],
        "He played the {X} of hearts, and she also played the"),
    "status tier": F(L_, "bronze silver gold platinum diamond", [
        "He upgraded to {X} status.", "The {X} level comes with extra rewards.", "Our company is a {X} sponsor of the event.",
        "She reached {X} status with the airline.", "The {X} tier costs more each month.", "He received a {X} award for his sales.",
        "The album was certified {X}.", "The {X} membership includes free shipping."],
        "She reached {X} status, and her husband also reached"),
    "credit rating": F(L_, "AAA AA A BBB BB B", [
        "The bond is rated {X}.", "The agency gave the company a {X} rating.", "Its credit rating fell to {X}.",
        "The city holds a {X} credit rating.", "Analysts rated the debt {X}.", "The fund only buys {X} bonds.",
        "The country's rating is {X}.", "Investors prefer {X} securities."],
        "The bond is rated {X}, and the loan is also rated"),
    # --- school and life
    "school year": F(L_, "freshman sophomore junior senior", [
        "She is a {X} in high school.", "He was a {X} at the university.", "The {X} class organized a dance.",
        "As a {X}, she joined the debate team.", "The {X} students took the exam.", "Every {X} must meet with an advisor.",
        "My brother is now a {X} in college.", "The {X} year was the hardest."],
        "She was a {X} in college, and her roommate was also a"),
    "letter grade": F(L_, "A B C D F", [
        "He received a grade of {X} in chemistry.", "My final grade was {X}.", "The essay earned a grade of {X}.",
        "Her report card showed {X} in history.", "The teacher gave the project a grade of {X}.",
        "Next to his name the teacher wrote {X}.", "His grade in English was {X}.", "She finished the course with {X}."],
        "Her grade in math was {X}, and her grade in history was also"),
    "school level": F(L_, "preschool kindergarten elementary college graduate", [
        "The {X} students went on a field trip.", "She teaches {X} classes.", "The {X} program has a long waiting list.",
        "He is now a teacher at the {X} level.", "The {X} curriculum was updated.", "Our {X} classroom is bright.",
        "The {X} building is next to the park.", "Parents met the {X} staff."],
        "She taught at the {X} level, and her sister also taught at the"),
    "degree": F(L_, "associate bachelor master doctoral", [
        "She finished her {X} degree last year.", "He is studying for his {X} degree.", "Applicants must hold the {X} degree.",
        "Her {X} degree is in biology.", "The {X} program takes several years.", "He completed the {X} program online.",
        "They offer the {X} program in nursing.", "My {X} thesis was about birds."],
        "She earned her {X} degree, and her brother also earned his"),
    "life stage": F(L_, "infant baby toddler child teenager adult senior", [
        "The {X} slept through the night.", "The doctor examined the {X} carefully.", "Every {X} needs good food and rest.",
        "The {X} was afraid of the dark.", "Our neighbor cares for the {X} next door.", "The {X} laughed at the joke.",
        "The tired {X} sat alone in the waiting room.", "The {X} needed a new pair of shoes."],
        "The nurse helped the {X}, and the doctor also helped the"),
    "generation": F(L_, "grandson son father grandfather", [
        "The {X} inherited the farm.", "The {X} told stories by the fire.", "Everyone at the party congratulated the {X}.",
        "The {X} built the house.", "The {X} was named after his uncle.", "The old photo shows the {X} as a boy.",
        "The {X} loved fishing.", "The {X} came to visit."],
        "The farm passed to the {X}, and the house also passed to the"),
    "decade of life": F(L_, "teens twenties thirties forties fifties sixties seventies eighties", [
        "She is in her {X} now.", "He started running in his {X}.", "Many people change careers in their {X}.",
        "My parents met in their {X}.", "He wrote his first novel in his {X}.", "She learned to swim in her {X}.",
        "Life feels different in your {X}.", "They bought a house in their {X}."],
        "She is in her {X}, and her husband is also in his"),
    "relationship": F(L_, "single dating engaged married divorced", [
        "She is {X} now.", "He has been {X} for two years.", "They were {X} when they moved here.",
        "My cousin is {X} and lives in Paris.", "Her status says {X}.", "They were already {X} by then.",
        "Are you still {X}?", "Being {X} has changed him."],
        "She is {X}, and her sister is also"),
    "closeness": F(L_, "stranger acquaintance colleague friend partner spouse", [
        "She trusted the {X} with her secret.", "He had dinner with the {X} last night.", "I asked the {X} for advice.",
        "The {X} remembered her birthday.", "He lent money to the {X}.", "She waved at the {X} across the street.",
        "The {X} called late at night.", "They talked with the {X} for hours."],
        "She called the {X}, and later she also called the"),
    "grief stage": F(L_, "denial anger bargaining depression acceptance", [
        "After the loss, she went through {X}.", "He was stuck in {X} for months.", "The therapist talked about {X}.",
        "Grief can include {X}.", "Her letters were full of {X}.", "The family moved slowly toward {X}.",
        "The book has a chapter on {X}.", "She recognized her own {X}."],
        "She felt {X}, and her brother also felt"),
    "insect stage": F(L_, "egg larva pupa adult", [
        "The {X} stage lasts two weeks.", "The butterfly stays in the {X} stage for days.",
        "Scientists studied the {X} under a microscope.", "The {X} was found on a leaf.", "The beetle {X} is hard to spot.",
        "Each {X} needs warmth to develop.", "The {X} stage is the most vulnerable.", "A bird ate the {X}."],
        "The scientist examined the {X}, and her student also examined the"),
    # --- sizes and amounts
    "clothing size": F(L_, "XS S M L XL XXL", [
        "I usually wear size {X}.", "The shirt only comes in size {X}.", "She ordered the jacket in size {X}.",
        "The store ran out of size {X}.", "He needs a size {X} in this brand.", "Size {X} fits me best.",
        "The label on the dress says {X}.", "Do you have this sweater in {X}?"],
        "She bought the shirt in size {X}, and he bought the jacket in size"),
    "coffee size": F(L_, "short tall grande venti", [
        "I'll have a {X} latte, please.", "She ordered a {X} coffee.", "He always gets the {X} size.",
        "One {X} cappuccino to go.", "The {X} cup costs more.", "Can I get a {X} mocha?", "A {X} tea, no sugar.",
        "She asked for a {X} americano."],
        "She ordered a {X} latte, and he also ordered a"),
    "coin": F(L_, "penny nickel dime quarter dollar", [
        "He found a {X} on the sidewalk.", "She dropped a {X} into the jar.", "I need a {X} for the parking meter.",
        "The {X} rolled under the couch.", "She gave the child a {X}.", "He flipped a {X} to decide.",
        "The old {X} was worth a lot to collectors.", "There was a {X} in his pocket."],
        "He found a {X} on the floor, and she also found a"),
    "settlement": F(L_, "hamlet village town city metropolis", [
        "They live in a {X} by the river.", "The {X} grew quickly after the railway came.", "She was born in a {X} in the north.",
        "The {X} elected a new mayor.", "Tourists rarely visit the {X}.", "The road leads to a {X} in the valley.",
        "The {X} was founded centuries ago.", "Everyone in the {X} knew the baker."],
        "She grew up in a {X}, and her husband also grew up in a"),
    "geographic scale": F(L_, "street neighborhood city county state country continent", [
        "She knows everyone in the {X}.", "The {X} passed a new law.", "Crime fell across the {X} last year.",
        "The {X} has many parks.", "He traveled across the whole {X}.", "The {X} map hung on the wall.",
        "News spread through the {X}.", "The {X} is home to many families."],
        "The news spread across the {X}, and the rumor also spread across the"),
    "water body": F(L_, "puddle pond lake sea ocean", [
        "The children threw stones into the {X}.", "A duck swam across the {X}.", "The {X} froze in winter.",
        "We looked out over the {X}.", "The {X} was calm this morning.", "Fish lived in the {X}.",
        "The {X} reflected the sky.", "He fell into the {X}."],
        "The boy swam in the {X}, and his sister also swam in the"),
    "rock size": F(L_, "pebble stone rock boulder", [
        "He kicked the {X} down the path.", "She picked up a {X} from the beach.", "A {X} blocked the trail.",
        "The {X} rolled down the hill.", "The child sat on the {X}.", "He threw the {X} into the river.",
        "The {X} was smooth and gray.", "There was a {X} in my shoe."],
        "He picked up the {X}, and she also picked up the"),
    "ensemble": F(L_, "solo duo trio quartet quintet sextet", [
        "They performed as a {X} at the festival.", "The {X} played jazz all night.", "She joined a {X} in college.",
        "The {X} released a new album.", "The concert featured a string {X}.", "Tickets for the {X} sold out.",
        "The {X} rehearsed every Tuesday.", "The famous {X} toured Europe."],
        "They played as a {X}, and later they played again as a"),
    "taxonomy rank": F(L_, "species genus family order class phylum kingdom", [
        "The animal belongs to a different {X}.", "Biologists placed the plant in a new {X}.", "Each {X} contains many organisms.",
        "The {X} name is written in Latin.", "Scientists classified the fossil by its {X}.",
        "This {X} includes both insects and spiders.", "The beetle was moved to another {X}.",
        "In biology, the {X} is a rank of classification."],
        "The beetle was placed in a new {X}, and the spider was also placed in a new"),
    "quantifier": F(L_, "none few some many most all", [
        "He said {X} of the students passed.", "We found that {X} of the apples were ripe.", "In the end, {X} of the guests stayed.",
        "She noticed that {X} of the lights were on.", "The survey showed that {X} of the workers agreed.",
        "By noon, {X} of the tickets were sold.", "He believed that {X} of the claims were true.",
        "After the storm, {X} of the trees were still standing."],
        "The teacher said that {X} of the students passed, and the principal agreed that"),
    "wealth": F(L_, "penniless poor comfortable wealthy", [
        "The family was {X} after the war.", "He grew up {X} in a small town.", "They became {X} in a few years.",
        "Her parents were {X} when she was born.", "The village was {X} for centuries.", "He died {X}.",
        "The {X} family lived on the hill.", "She remained {X} her whole life."],
        "His family was {X}, and her family was also"),
    # --- weather, nature, matter
    "wind": F(L_, "calm breeze wind gale storm hurricane", [
        "The forecast predicts a {X} tomorrow.", "The sailors noticed the {X} at dawn.", "We talked about the {X} all evening.",
        "The {X} came in from the sea.", "The {X} lasted until morning.", "Nobody expected the {X} that night.",
        "The radio warned of a {X} near the coast.", "The fishermen watched the {X} from the harbor."],
        "The captain reported a {X}, and the pilot also reported a"),
    "precipitation": F(L_, "mist drizzle shower rain downpour deluge", [
        "The {X} lasted all afternoon.", "We walked home through the {X}.", "The {X} soaked the garden.",
        "The {X} began just after noon.", "The forecast mentioned the {X}.", "The {X} stopped before dinner.",
        "Children played in the {X}.", "The {X} fell on the city all night."],
        "The farmer watched the {X}, and the children also watched the"),
    "state of matter": F(L_, "solid liquid gas plasma", [
        "The substance is a {X} at room temperature.", "Heat turned the material into a {X}.",
        "In this experiment the sample is a {X}.", "The {X} phase is the most common here.", "Scientists studied the {X} in the lab.",
        "The element exists as a {X} on the star.", "The chemist described it as a {X}.", "The {X} state has different properties."],
        "At this temperature it is a {X}, and at that temperature it is also a"),
    "moon phase": F(L_, "new crescent half gibbous full", [
        "Tonight there is a {X} moon.", "The {X} moon rose over the hills.", "We camped under a {X} moon.",
        "The calendar marks the {X} moon.", "A {X} moon hung in the sky.", "By the light of the {X} moon, they walked home.",
        "The {X} moon was hidden by clouds.", "Farmers planted by the {X} moon."],
        "Last night there was a {X} moon, and tonight there is also a"),
    "brightness": F(L_, "dark dim bright brilliant", [
        "The room was {X}.", "The light in the hall was {X}.", "Her screen was too {X}.", "The sky turned {X}.",
        "The stage looked {X}.", "The lamp is {X} tonight.", "His office was always {X}.", "The kitchen felt {X} in the morning."],
        "The room was {X}, and the hall was also"),
    # --- time and sequence
    "meal": F(L_, "breakfast brunch lunch dinner supper", [
        "We ate {X} on the porch.", "She cooked {X} for the whole family.", "He skipped {X} today.",
        "Let's meet for {X} tomorrow.", "What should we make for {X}?", "The kids were late for {X}.",
        "We talked over {X} about the trip.", "The hotel serves {X} in the garden."],
        "They had {X} at the cafe, and the next day they had"),
    "historical era": F(L_, "prehistoric ancient classical medieval renaissance modern contemporary futuristic", [
        "The museum has a room of {X} art.", "She studies {X} history.", "The building shows {X} design.",
        "They found {X} tools in the cave.", "The film is set in {X} times.", "He loves {X} architecture.",
        "The book describes {X} warfare.", "The exhibit features {X} sculptures."],
        "The museum showed {X} art, and the gallery also showed"),
    "schedule": F(L_, "hourly daily weekly monthly quarterly annually", [
        "The report is published {X}.", "We meet {X} to review progress.", "The data is updated {X}.",
        "Payments are made {X}.", "The machine is cleaned {X}.", "The newsletter comes out {X}.",
        "Prices are adjusted {X}.", "The team checks the system {X}."],
        "The report is published {X}, and the newsletter is also published"),
    "tournament round": F(L_, "qualifier quarterfinal semifinal final", [
        "The team lost in the {X}.", "She won her {X} easily.", "The {X} will be played on Sunday.",
        "Tickets for the {X} sold out.", "He scored twice in the {X}.", "The {X} was broadcast live.",
        "They reached the {X} last year.", "Rain delayed the {X}."],
        "The team played well in the {X}, and they also played well in the"),
    "project stage": F(L_, "plan design build test launch", [
        "The team will {X} the new app next month.", "We need to {X} the system first.", "Their job is to {X} the product.",
        "They hired experts to {X} the bridge.", "It took a year to {X} the rocket.", "Our group will {X} the website.",
        "She was asked to {X} the software.", "The company plans to {X} the satellite."],
        "The team will {X} the app, and next year they will also"),
    "sleep": F(L_, "asleep drowsy awake alert", [
        "The baby was {X}.", "By midnight he was {X}.", "She felt {X} during the lecture.", "The guard stayed {X} all night.",
        "The patient is {X} now.", "The children were {X} in the car.", "He seemed {X} at breakfast.", "The dog was {X} by the fire."],
        "The baby was {X}, and the dog was also"),
    # --- degree and likelihood
    "frequency": F(L_, "never rarely seldom sometimes often usually always", [
        "She {X} eats breakfast.", "He {X} calls his parents.", "We {X} go to the beach.", "They {X} walk to work.",
        "I {X} read before bed.", "The bus {X} arrives on time.", "My cat {X} sleeps on the bed.", "Our teacher {X} gives homework."],
        WORD_COPY),
    "probability": F(L_, "impossible unlikely possible likely certain", [
        "It is {X} that it will rain tomorrow.", "The doctor said a full recovery is {X}.", "Success seemed {X} after the first test.",
        "A delay is {X} at this point.", "Experts think a deal is {X}.", "It was {X} that they would win.",
        "Another storm is {X} this week.", "The outcome is {X}, according to the report."],
        "The doctor said recovery was {X}, and the nurse also said it was"),
    "hunger": F(L_, "starving hungry peckish satisfied full stuffed", [
        "After the hike, I was {X}.", "The children were {X} by noon.", "He felt {X} after lunch.", "She was {X} when she got home.",
        "Are you {X}?", "We were all {X} at the end of the meal.", "The puppy seemed {X}.", "By evening he was {X}."],
        "She said she was {X}, and he said he was also"),
    "anger": F(L_, "irritated annoyed angry furious enraged", [
        "She was {X} about the delay.", "He felt {X} when he saw the bill.", "The customer seemed {X}.",
        "My boss was {X} with me.", "They were {X} at the decision.", "The fans grew {X}.", "His voice sounded {X}.",
        "She looked {X} after the call."],
        "She was {X}, and her husband was also"),
    "quality": F(L_, "terrible bad mediocre decent good great excellent", [
        "The food was {X}.", "The movie was {X}.", "Her performance was {X}.", "The service at the hotel was {X}.",
        "His work this year has been {X}.", "The reviews were {X}.", "The weather was {X}.", "The concert was {X}."],
        "The food was {X}, and the service was also"),
    # --- music
    "tempo": F(L_, "largo adagio andante moderato allegro presto", [
        "The movement is marked {X}.", "Play this section {X}.", "The {X} movement was beautiful.",
        "The orchestra played the {X} beautifully.", "The tempo marking says {X}.", "She practiced the {X} every day.",
        "The {X} section was the hardest.", "The conductor chose a true {X}."],
        "The first movement is marked {X}, and the second movement is also marked"),
    "dynamics": F(L_, "pianissimo piano forte fortissimo", [
        "The passage is marked {X}.", "Play this part {X}.", "The choir sang {X}.", "The score says {X} here.",
        "The music suddenly turns {X}.", "The violins enter {X}.", "End the piece {X}.", "The marking {X} appears in bar ten."],
        "The passage is marked {X}, and the next passage is also marked"),
    "karate belt": F(L_, "white yellow orange green blue brown black", [
        "She earned her {X} belt last month.", "He is training for his {X} belt.", "The {X} belt students practiced kicks.",
        "Only {X} belt students can spar.", "The test for the {X} belt is hard.", "My son just got his {X} belt.",
        "She tied her new {X} belt.", "The instructor's {X} belt is old."],
        "She earned her {X} belt, and her brother also earned his"),
    # --- sequences learned by rote
    "greek letter": F(L_, "alpha beta gamma delta epsilon zeta eta theta", [
        "The {X} particle was detected.", "Label the angle {X}.", "The {X} version of the software is ready.",
        "In the equation, {X} stands for a constant.", "The {X} wave was strong.", "She belongs to the {X} chapter.",
        "We measured the {X} value.", "The {X} variant spread quickly."],
        "The first symbol was {X}, and the second symbol was also"),
    "NATO alphabet": F(L_, "Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel", [
        "The pilot said {X} over the radio.", "Team {X} will move first.", "Unit {X}, please respond.",
        "Squad {X} reached the bridge.", "The code word was {X}.", "Company {X} held the hill.",
        "Report to checkpoint {X}.", "This is {X} calling base."],
        "The code word was {X}, and the reply was also"),
    # --- cycles
    "zodiac": F(C_, "Aries Taurus Gemini Cancer Leo Virgo Libra Scorpio Sagittarius Capricorn Aquarius Pisces", [
        "Her star sign is {X}.", "His horoscope for {X} predicted luck.", "People born under {X} are said to be stubborn.",
        "The sun is in {X} this month.", "My sign is {X}.", "Today's reading for {X} mentions travel.",
        "The astrologer described {X} in detail.", "Her birthday falls under {X}."],
        "Her star sign is {X}, and his star sign is also"),
    "Chinese zodiac": F(C_, "Rat Ox Tiger Rabbit Dragon Snake Horse Goat Monkey Rooster Dog Pig", [
        "She was born in the Year of the {X}.", "The Year of the {X} begins in February.",
        "He celebrated the Year of the {X} with his family.", "Children born in the Year of the {X} are said to be lucky.",
        "The calendar shows the Year of the {X}.", "Next year is the Year of the {X}.",
        "The festival marked the Year of the {X}.", "In the Chinese zodiac, the {X} is a sign of strength."],
        "She was born in the Year of the {X}, and her cousin was also born in the Year of the"),
    "solfege": F(C_, "do re mi fa sol la ti", [
        "The choir sang the note {X}.", "Sing {X} and hold it.", "The next note is {X}.", "She hit a perfect {X}.",
        "Start on {X} and go up.", "The melody ends on {X}.", "He played {X} on the recorder.",
        "The teacher pointed to {X} on the chart."],
        "The first note was {X}, and the last note was also"),
    "musical note": F(C_, "C D E F G A B", [
        "The song is in the key of {X}.", "She played the note {X} on the piano.", "Tune the string to {X}.",
        "The melody starts on {X}.", "The symphony is in {X} major.", "He sang a high {X}.", "The chord is built on {X}.",
        "Move your finger to {X}."],
        "The song is in the key of {X}, and the next song is also in the key of",
        alt={"circle of fifths": "C G D A E B F"}),
}
# Known structures, as calibration for the shape tests (circles we know, scales we called straight or curved before).
import core as _core
_SV = dict(
    howsoon=("someday eventually later soon shortly immediately now", [
        "We will fix the problem {X}.", "They promised to call {X}.", "The results will arrive {X}.", "I will finish the report {X}.",
        "She plans to move abroad {X}.", "The bridge will reopen {X}.", "He said he would apologize {X}.",
        "The company will announce the decision {X}."], "She said they would leave {X}, and he agreed they would leave"),
    speed=("sluggish slow moderate steady brisk fast rapid", [
        "The car moved at a {X} pace.", "The project made {X} progress.", "The river had a {X} current.", "He walked at a {X} speed.",
        "The economy saw {X} growth.", "The train kept a {X} pace.", "The dancers moved in {X} steps.", "The recovery was {X}."],
        "The car moved at a {X} pace, and the bus also moved at a"))
CONTROLS = {
    "(weekday)": F(C_, " ".join(_core.FAMILIES["weekday"]["vals"]), _core.FAMILIES["weekday"]["templates"][:8],
                   "I'll see you on {X}, and she will also see you on"),
    "(month)": F(C_, " ".join(_core.FAMILIES["month"]["vals"]), _core.FAMILIES["month"]["templates"][:8],
                 "My birthday is in {X}, and her birthday is also in"),
    "(number)": F(L_, "one two three four five six seven eight nine", _core.FAMILIES["number"]["templates"][:8],
                  "I bought {X} apples, and she also bought"),
    "(temperature)": F(L_, " ".join(_core.FAMILIES["temperature"]["vals"]), _core.FAMILIES["temperature"]["templates"][:8],
                       "The water was {X}, and the air was also"),
    "(size)": F(L_, " ".join(_core.FAMILIES["size"]["vals"]), _core.FAMILIES["size"]["templates"][:8],
                "They live in a {X} house, and we also live in a"),
    "(how soon)": F(L_, *_SV["howsoon"]),
    "(speed)": F(L_, *_SV["speed"]),
}
FAMILIES.update(CONTROLS)
ALPHAS = np.linspace(0, 1, 13)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def partial(a, b, c):
    """Rank correlation of a and b with c partialled out."""
    ra, rb, rc = (np.argsort(np.argsort(x)).astype(float) for x in (a, b, c))
    res = lambda y: y - np.polyval(np.polyfit(rc, y, 1), rc)
    ea, eb = res(ra), res(rb)
    return float(np.corrcoef(ea, eb)[0, 1]) if ea.std() > 1e-9 and eb.std() > 1e-9 else 0.0


def cyc_score(xy, order_idx):
    """Neighbours (in the hypothesised cyclic order) that are also neighbours in angle around the centre."""
    n = len(order_idx); c = xy - xy.mean(0); ang = np.arctan2(c[:, 1], c[:, 0])
    rank = np.empty(n, int); rank[np.argsort(ang)] = np.arange(n)
    return int(sum((rank[order_idx[(i + 1) % n]] - rank[order_idx[i]]) % n in (1, n - 1) for i in range(n)))


def order_stats(mu, order, perm_n=4000, seed=0):
    """mu: values in hypothesised order. Linear: rank corr on PC1 (and best of PC1-3). Cyclic: adjacency on PC1-2 plane
    (and best plane of PC1-3). Permutation p-values for both."""
    V = len(mu); X = mu - mu.mean(0); _, S, Vt = np.linalg.svd(X, full_matrices=False); P = X @ Vt[:3].T
    ev = (S ** 2 / (S ** 2).sum())[:3]
    rng = np.random.default_rng(seed); idx = np.arange(V)
    if order == "linear":
        obs = [spearman(P[:, k], idx) for k in range(3)]
        n1, nb = [], []
        for _ in range(perm_n):
            p = rng.permutation(V); r = [abs(spearman(P[:, k], p)) for k in range(3)]; n1.append(r[0]); nb.append(max(r))
        best = int(np.argmax(np.abs(obs)))
        def bend(o):
            r = np.argsort(np.argsort(o)).astype(float); q = (r - r.mean()) ** 2
            q = q - np.polyval(np.polyfit(r, q, 1), r); q /= np.linalg.norm(q) + 1e-12
            return max(abs(float(np.corrcoef(P[:, k], q)[0, 1])) for k in (1, 2)) if V >= 4 else 0.0
        hs = bend(idx); hnull = [bend(rng.permutation(V)) for _ in range(1000)]
        return dict(horseshoe=hs, p_horseshoe=float(np.mean(np.array(hnull) >= hs - 1e-9)), rho=obs[0], p=float(np.mean(np.array(n1) >= abs(obs[0]) - 1e-9)), best_axis=best + 1, rho_best=obs[best],
                    p_best=float(np.mean(np.array(nb) >= abs(obs[best]) - 1e-9)), ev=ev.round(3).tolist(), pcs=P)
    planes = [(0, 1), (0, 2), (1, 2)]
    obs = [cyc_score(P[:, list(pl)], idx) for pl in planes]
    n1, nb = [], []
    for _ in range(perm_n):
        p = rng.permutation(V); s = [cyc_score(P[:, list(pl)], p) for pl in planes]; n1.append(s[0]); nb.append(max(s))
    best = int(np.argmax(obs))
    return dict(score=obs[0], of=V, p=float(np.mean(np.array(n1) >= obs[0])), best_plane=planes[best], score_best=obs[best],
                p_best=float(np.mean(np.array(nb) >= obs[best])), ev=ev.round(3).tolist(), pcs=P)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    model_dir, out = args[0], args[1]
    LAYERS = [int(x) for x in (args[2] if len(args) > 2 else "6,12").split(",")]
    steer = "--no-steer" not in sys.argv
    cache = sys.argv[sys.argv.index("--cache") + 1] if "--cache" in sys.argv else None
    only = os.environ.get("ONLY")
    fams = {k: v for k, v in FAMILIES.items() if not only or k in only.split(",")}
    torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count())))
    os.makedirs(out, exist_ok=True)
    from wordfreq import zipf_frequency
    tok = AutoTokenizer.from_pretrained(model_dir, cache_dir=cache)
    m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, cache_dir=cache).eval()
    lm = m.model.language_model if hasattr(m.model, "language_model") else m.model
    U = m.get_output_embeddings().weight.detach()
    cap, st = {}, dict(delta=None, pos=None, layer=None)

    def hook(l):
        def f(mod, inp, outp):
            h = outp[0] if isinstance(outp, tuple) else outp
            if st["delta"] is not None and st["layer"] == l:
                h = h.clone(); h[:, st["pos"]] += st["delta"]
                outp = (h,) + tuple(outp[1:]) if isinstance(outp, tuple) else h
            cap[l] = h.detach()
            return outp
        return f
    for l in LAYERS: lm.layers[l].register_forward_hook(hook(l))
    enc = lambda s: tok(s, add_special_tokens=False).input_ids

    def locate(tpl, v):
        """Token ids of the filled template and the position of the value's last token (None if the split shifts)."""
        pre = tpl[:tpl.index("{X}")]
        ids, ip, iv = enc(tpl.replace("{X}", v)), enc(pre.rstrip()), enc(" " + v)
        if not pre.endswith(" ") or ids[:len(ip) + len(iv)] != ip + iv: return None, None
        return ids, len(ip) + len(iv) - 1

    res = {}
    for name, Fm in fams.items():
        vals = Fm["vals"]; V = len(vals)
        first = [enc(" " + v)[0] for v in vals]
        R = {L: np.zeros((len(Fm["templates"]), V, lm.config.hidden_size), np.float32) for L in LAYERS}
        keep = []
        for ti, tpl in enumerate(Fm["templates"]):
            loc = [locate(tpl, v) for v in vals]
            if any(p is None for _, p in loc): print(f"  {name}: skipped template (token split shifts): {tpl}"); continue
            keep.append(ti)
            by_len = {}
            for vi, (ids, p) in enumerate(loc): by_len.setdefault(len(ids), []).append((vi, ids, p))
            for rows in by_len.values():
                with torch.inference_mode(): lm(input_ids=torch.tensor([r[1] for r in rows]), use_cache=False)
                for L in LAYERS:
                    for j, (vi, _, p) in enumerate(rows): R[L][ti, vi] = cap[L][j, p].float().numpy()
        freq = np.array([zipf_frequency(v.lower() if v.isupper() and len(v) > 2 else v, "en") for v in vals])
        ntok = [len(enc(" " + v)) for v in vals]
        orders = {"main": np.arange(V)}
        for an, seq in Fm.get("alt", {}).items(): orders[an] = np.array([vals.index(x) for x in seq.split()])
        res[name] = dict(order=Fm["order"], vals=vals, n_tokens=ntok, zipf=freq.round(2).tolist(), templates=len(keep), layers={})
        for L in LAYERS:
            Rk = R[L][keep]; mu = (Rk - Rk.mean(1, keepdims=True)).mean(0)
            np.save(f"{out}/{name.replace(' ', '_')}_L{L}_means.npy", mu.astype(np.float16))
            row = {}
            for on, oi in orders.items():
                s = order_stats(mu[oi], Fm["order"])
                if Fm["order"] == "linear":
                    s["rho_freq"] = spearman(s["pcs"][:, 0], freq[oi]); s["order_vs_freq"] = spearman(np.arange(V), freq[oi])
                    s["rho_partial"] = partial(s["pcs"][:, 0], np.arange(V), freq[oi])
                    # distances grow with order gap?
                    D = np.linalg.norm(mu[oi][:, None] - mu[oi][None], axis=-1); iu = np.triu_indices(V, 1)
                    s["gap_rho"] = spearman(D[iu], (iu[1] - iu[0]).astype(float))
                s["pcs"] = s["pcs"][:, :2].round(3).tolist()
                row[on] = s
            # 2-D layout of the means for the report (PC1-2 of the template-centred means)
            if steer and len(set(first)) == V:
                walks, nodes = [], []
                tgt = V - 1 if Fm["order"] == "linear" else V // 2
                for s_i, t_i in ((0, tgt), (tgt, 0)):
                    ids, p = locate(Fm["copy"], vals[s_i])
                    if ids is None: break
                    st["pos"], st["layer"] = p, L
                    for kind, D in (("chord", np.stack([a * (mu[t_i] - mu[s_i]) for a in ALPHAS])),
                                    ("node", np.stack([mu[v] - mu[s_i] for v in range(V)]))):
                        st["delta"] = torch.tensor(D, dtype=torch.float32)
                        with torch.inference_mode():
                            h = lm(input_ids=torch.tensor([ids] * len(D)), use_cache=False).last_hidden_state[:, -1]
                        st["delta"] = None
                        q = torch.softmax(h @ U.T, -1)[:, first].numpy()
                        rec = dict(start=vals[s_i], idx=q.argmax(1).tolist(), mass=q.sum(1).round(3).tolist(), q=q / q.sum(1, keepdims=True))
                        (walks if kind == "chord" else nodes).append(rec)
                if len(walks) == 2:
                    inner = list(range(1, tgt))
                    reach = [v for v in inner if nodes[0]["idx"][v] == v and nodes[1]["idx"][v] == v]
                    on_chord = set(walks[0]["idx"]) | set(walks[1]["idx"])
                    # graded: best share a middle word gets anywhere on the chord, relative to its share at its own average
                    rel = [max(walks[k]["q"][:, v].max() / max(nodes[k]["q"][v, v], 1e-9) for k in (0, 1)) for v in inner]
                    row["steer"] = dict(rel=[round(float(x), 3) for x in rel], rel_median=float(np.median(rel)) if rel else None,
                        node_rate=float(np.mean([nodes[k]["idx"][v] == v for k in (0, 1) for v in inner])) if inner else None,
                        reach=[vals[v] for v in reach], visited=[vals[v] for v in inner if v in on_chord],
                        chord_cover=(sum(v in on_chord for v in reach) / len(reach)) if reach else None,
                        walks=[dict(start=w["start"], words=[vals[i] for i in w["idx"]], mass=w["mass"]) for w in walks],
                        nodes=[dict(start=n["start"], words=[vals[i] for i in n["idx"]]) for n in nodes],
                        mass=float(np.mean([np.mean(w["mass"]) for w in walks])))
            elif steer:
                row["steer"] = dict(skipped="first tokens not distinct")
            res[name]["layers"][L] = row
            o = row["main"]
            if Fm["order"] == "linear":
                txt = (f"rank corr {o['rho']:+.2f} (p {o['p']:.3f}) | best axis PC{o['best_axis']} {o['rho_best']:+.2f} (p {o['p_best']:.3f})"
                       f" | freq: PC1~freq {o['rho_freq']:+.2f}, order~freq {o['order_vs_freq']:+.2f}, partial {o['rho_partial']:+.2f}"
                       f" | gap~dist {o['gap_rho']:+.2f} | bend {o['horseshoe']:.2f} (p {o['p_horseshoe']:.3f})")
            else:
                txt = f"circle {o['score']}/{o['of']} (p {o['p']:.3f}) | best plane {o['best_plane']} {o['score_best']}/{o['of']} (p {o['p_best']:.3f})"
                for an in orders:
                    if an != "main": txt += f" | {an}: {row[an]['score']}/{row[an]['of']} (p {row[an]['p']:.3f})"
            sv = row.get("steer", {})
            if "node_rate" in sv:
                txt += (f"\n        node check {sv['node_rate']:.0%} of middle words; reproducible {len(sv['reach'])}/{V - 2}; "
                        f"chord passes {sv['chord_cover'] if sv['chord_cover'] is None else round(sv['chord_cover'], 2)} of them; "
                        f"middle words' best share on chord / at own node: median {sv['rel_median']:.2f}; "
                        f"chord {' → '.join(dedup(sv['walks'][0]['words']))}  (mass {sv['mass']:.2f})")
            print(f"{name:17s} L{L:2d} {txt}", flush=True)
        json.dump(res, open(f"{out}/family_sweep.json", "w"), indent=1)
    os._exit(0)


def dedup(ws):
    return [ws[0]] + [w for i, w in enumerate(ws[1:], 1) if w != ws[i - 1]]


if __name__ == "__main__":
    main()
