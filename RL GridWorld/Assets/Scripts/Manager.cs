using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class Manager : MonoBehaviour
{
    public enum Algorithm
    {
        Q_Learning,
        DynaQ,
        DynaQPlus,
    }

    // Train Parameters
    [Header("Run Settings")]
    [SerializeField] private int numRuns;
    [SerializeField] private int numEpisodes;
    [SerializeField] private bool useStandardStart;

    [Header("Agent Settings")]
    // Agent Parameters
    [SerializeField] private string Name;
    [SerializeField] private int numActions;
    [SerializeField] private int numStates;
    [SerializeField] private float epsilon;
    [SerializeField] private float gamma;
    [SerializeField] private float alpha;
    [SerializeField] private Rigidbody agentRb;

    [Header("Algorithm Settings")]
    [SerializeField] private Algorithm algo;
    [SerializeField] private float kappa;
    [SerializeField] private int planningSteps;

    [Header("Environment Settings")]
    [SerializeField] private List<GameObject> miscList;
    [SerializeField] private List<GameObject> openingDoors;
    [SerializeField] private List<GameObject> closingDoors;

 

    // Visualization Parameters
    [SerializeField] private GameObject arrow_white;
    [SerializeField] private GameObject arrow_blue;
    private List<GameObject> arrowWhiteList = new List<GameObject>();
    private List<GameObject> arrowBlueList = new List<GameObject>();
    private List<int> resultActions = new List<int>();

    private RL_Agent agent;

    // Position Trackers
    private Vector3 startPos;

    // Retrun Params
    private float reward = 0;
    private int state;
    private bool terminal = false;

    // Previous action/state
    private int prevAction;
    private int prevState;

    // Planning Parameters
    private List<int> vistedStates = new List<int>();
    private Dictionary<int, (int, float)> model = new Dictionary<int, (int, float)>();
    private List<int> tauList = new List<int>();
    

    // Helper Flags
    private bool hasCollided = false;
    private bool trained = false;
    private bool started = false;


    // Performance Trackers
    private List<float> allRewards = new List<float>();
    private List<float> allQ = new List<float>();

    // Debug Parameters
    private int obstacle_counter = 0;
    private List<int> obstacleStateIdx = new List<int>() { 75,76,77,78,79};
    private List<float> debugList = new List<float>();

    private void Awake()
    {
        Physics.autoSimulation = false;

        // Spread the arrows over the floor
        for (int h=0; h<19; h++)
        {
            for (int w=0; w<9; w++)
            {
                var newArrowW = Instantiate(arrow_white);
                var newArrowB = Instantiate(arrow_blue);
                newArrowW.transform.position = new Vector3(w + 1f, -0.45f, h + 1f);
                newArrowB.transform.position = new Vector3(w + 1f, -0.4f, h + 1f);

                arrowWhiteList.Add(newArrowW);
                arrowBlueList.Add(newArrowB);
            }
        }

    }

    // Start is called before the first frame update
    void Start()
    {
        startPos = agentRb.position;

        // Initialize Variables
        for (int i = 0; i < numStates; i++)
        {
            for (int j = 0; j < numActions; j++)
            {
                allQ.Add(0f);
                tauList.Add(0);
            }
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.W))
        {
            EnvStep(2);
            print(GetState());
        }
        else if (Input.GetKeyDown(KeyCode.S))
        {
            EnvStep(3);
            print(GetState());
        }
        else if (Input.GetKeyDown(KeyCode.D))
        {
            EnvStep(1);
            print(GetState());
        }
        else if (Input.GetKeyDown(KeyCode.A))
        {
            EnvStep(0);
            print(GetState());
        }
        else if (Input.GetKeyDown(KeyCode.Q))
        {
            Debug.Log("Started Analysis");
            started = true;
        }
        else if (Input.GetKeyDown(KeyCode.E)) { visualizeResultPath(); }

        else if (Input.GetKeyDown(KeyCode.B)) { print(_getModelValue()); }

    }

    private void FixedUpdate()
    {
        if (!trained && started)
        {
            trainingLoop();
            trained = true;
        }
        else if (!started)
        {
            Physics.Simulate(Time.fixedDeltaTime);
        }
    }

    private void trainingLoop()
    {
        for (int run = 0; run < numRuns; run++)
        {

            EnvInit();

            for (int episode = 0; episode < numEpisodes; episode++)
            {
                // Start Episode
                (reward, state, terminal) = EnvStart();
                var action = agent.agentStart(state);

                prevState = state;
                prevAction = action;
                
                // Step Thorugh Episode until termination
                while (true)
                {

                    (reward, state, terminal) = EnvStep(action);

                    if (algo == Algorithm.DynaQPlus)
                    {
                        for (int i = 0; i < tauList.Count; i++)
                        {
                            if (i != prevState * numActions + action) { tauList[i] += 1; }
                            else { tauList[i] = 0; }
                        }
                    }


                    prevState = state;
                    prevAction = action;




                    if (!terminal)
                    {
                        action = agent.agentStep(reward, state);
                        
                    }
                    else
                    {
                        allRewards[run] += agent.agenEnd(reward);
                        break;
                    }

                    // Planning
                    if(algo != Algorithm.Q_Learning)
                    {
                        if (episode > 0) { EnvPlanning(planningSteps); }
                    }
                    

                    // Open the gates
                    if (episode == (int)numEpisodes / 2) {
                        foreach (GameObject opening in openingDoors) { opening.SetActive(false); }
                        foreach (GameObject closing in closingDoors) { closing.SetActive(true); }
                    }

                }
            }

            // Get average q_values
            allQ = allQ.Zip<float,float,float>(agent.GetQ(), (x, y) => x + y).ToList();
        }
        resultActions = agent.VisualizeGreedyPolicy(arrowWhiteList, allQ, true);
        resultActions = agent.VisualizeGreedyPolicy(arrowBlueList, allQ, false);

        //foreach (float value in allRewards) { print(value); }
    }

    private void EnvInit() {
        allRewards.Add(0);
        agent = new RL_Agent(Name, numActions, numStates, epsilon, gamma, alpha);

        // Initialize Gameobject 
        foreach (GameObject opening in openingDoors) { opening.SetActive(true); }
        foreach (GameObject closing in closingDoors) { closing.SetActive(false); }

        if (algo == Algorithm.DynaQPlus)
        {
            for (int i = 0; i < tauList.Count; i++)
            {
                tauList[i] = 0;
            }
        }
    }

    public (float, int, bool) EnvStart()
    {
        reward = 0;
        terminal = false;

        agentRb.position = GetRandomStartPos(useStandardStart);
        Physics.Simulate(Time.fixedDeltaTime);

        // Set misc items to true
        foreach(GameObject obj in miscList) { obj.SetActive(true); }

        return (reward, GetState(), false);
    }

    public (float, int, bool) EnvStep(int action)
    {
        reward = 0;
        terminal = false;
        var prevPos = agentRb.position;

        switch (action)
        {
            case 0:
                agentRb.position += Vector3.forward;
                break;
            case 1:
                agentRb.position += Vector3.back;
                break;
            case 2:
                agentRb.position += Vector3.right;
                break;
            case 3:
                agentRb.position += Vector3.left;
                break;
            case 4:
                agentRb.position += new Vector3(1f,0,1f);
                break;
            case 5:
                agentRb.position += new Vector3(1f, 0, -1f);
                break;
            case 6:
                agentRb.position += new Vector3(-1f, 0, -1f);
                break;
            case 7:
                agentRb.position += new Vector3(-1f, 0, 1f);
                break;
        }

        Physics.Simulate(Time.fixedDeltaTime);
        if (hasCollided || terminal)
        {
            agentRb.position = prevPos;
            Physics.Simulate(Time.fixedDeltaTime);
            hasCollided = false;
        }


        if (!terminal) { reward -= 1; }

        // Store value for Model
        if (!vistedStates.Contains(prevState)) { vistedStates.Add(prevState); }

        UpdateModel(prevState, action, GetState(), reward);

        return (reward, GetState(), terminal);
    }

    // Planning

    private void EnvPlanning(int steps)
    {
        for (int i=0; i<steps; i++)
        {
            (var oldState, var oldAction, var newState, var reward) = _getModelValue();

            if (algo == Algorithm.DynaQPlus)
            {
                reward += kappa * Mathf.Sqrt(tauList[oldState * numActions + oldAction]);
            }

            agent.agentPlanning(oldState, oldAction, newState, reward);
        }
    }


    public int GetState()
    {
        var position = agentRb.position - Vector3.one;
        var index = (int)(9 * (position.z) + position.x);
        return index;
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.layer == 7)
        {
            terminal = true;
            Debug.LogWarning("FINISH LINE!!");
        }
        else if (other.gameObject.layer == 6)
        {
            terminal = true;
            reward -= 100;
        }
        else if (other.gameObject.layer == 8)
        {
            hasCollided = true;
        }
        else if (other.gameObject.layer == 9)
        {
            reward -= 7;
            //other.gameObject.SetActive(false);
        }
        else if (other.gameObject.layer == 10)
        {
            reward -= 10;
        }
    }


    private void visualizeResultPath()
    {
        foreach(GameObject arrow in arrowBlueList) { arrow.SetActive(false); }

        agentRb.position = GetRandomStartPos(useStandardStart);

        terminal = false;
        var it = 0;
        while (it < 100){

            var cState = GetState();
            var prevPos = agentRb.position;

            arrowBlueList[cState].SetActive(true);

            switch (resultActions[cState])
            {
                case 0:
                    agentRb.position += Vector3.forward;
                    break;
                case 1:
                    agentRb.position += Vector3.back;
                    break;
                case 2:
                    agentRb.position += Vector3.right;
                    break;
                case 3:
                    agentRb.position += Vector3.left;
                    break;
                case 4:
                    agentRb.position += new Vector3(1f, 0, 1f);
                    break;
                case 5:
                    agentRb.position += new Vector3(1f, 0, -1f);
                    break;
                case 6:
                    agentRb.position += new Vector3(-1f, 0, -1f);
                    break;
                case 7:
                    agentRb.position += new Vector3(-1f, 0, 1f);
                    break;
            }

            Physics.Simulate(Time.fixedDeltaTime);
            if (hasCollided || terminal)
            {
                agentRb.position = prevPos;
                Physics.Simulate(Time.fixedDeltaTime);
                hasCollided = false;
            }

            it++;
        }
    }


    private void UpdateModel(int old_state, int old_action ,int new_state, float reward)
    {
        var state = old_state * numActions + old_action;
        //print("STATE : " + old_state+"        ACTION : " +old_action+"         KEY : " + (old_state * numActions + old_action));
        if (model.ContainsKey(state)) { model[state] = (new_state, reward); }
        else { model.Add(state, (new_state, reward)); }
    }

    private (int,int,int, float) _getModelValue()
    {
        var stateIndex = Random.Range(0, vistedStates.Count);
        var state = vistedStates[stateIndex];

        // Check which actions were performed on that state
        var possibleActions = new List<int>();
        for (int j=0; j<numActions; j++)
        {
            if(model.ContainsKey(state*numActions + j)) { possibleActions.Add(j); }
        }
        var actionIndex = Random.Range(0, possibleActions.Count);
        var action = possibleActions[actionIndex];
        (var newState, var reward) = model[state * numActions + action];

        return (state,action,newState,reward);
    }

    private Vector3 GetRandomStartPos(bool useStandardStart)
    {
        if (useStandardStart) { return startPos; }

        var x = Random.Range(1, 9);
        var z = Random.Range(1, 3);

        return new Vector3(x, 0.5f, z);
    }
}
