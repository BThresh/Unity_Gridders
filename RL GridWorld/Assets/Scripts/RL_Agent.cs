using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class RL_Agent 
{
    private string name;
    private int numActions;
    private int numStates;
    private float epsilon;
    private float gamma;
    private float alpha;
    private float rewardSum;
    private List<float> q = new List<float>();


    // Prev Stats
    private int lastState;
    private int lastAction;

    // Debug Parameters
    public int obstacleCounter = 0;

    public RL_Agent(string name, int num_actions, int num_states, float epsilon, float gamma, float alpha)
    {
        this.name = name;
        numActions = num_actions;
        numStates = num_states;
        this.epsilon = epsilon;
        this.gamma = gamma;
        this.alpha = alpha;
        rewardSum = 0;

        for (int i = 0; i<numStates; i++)
        {
            for (int j=0; j<numActions; j++)
            {
                q.Add(0f);
            }
        }
    }


    public int agentStart(int state)
    {
        rewardSum = 0;

        int action = policy(state);

        lastState = state;
        lastAction = action;

        return action;
    }

    ///// AGENT STEP
    
    public int agentStep(float reward, int state)
    {
        var action = policy(state);
        rewardSum += reward;
        var oldQ = getQValues(lastState, lastAction);
        var current_q = getQValues(state);
        var maxQ = current_q.Max();

        QUpdate(maxQ, oldQ, reward, false);

        lastState = state;
        lastAction = action;

        return action;
    }

    public float agenEnd(float reward)
    {
        rewardSum += reward;

        var oldQ = getQValues(lastState, lastAction);

        QUpdate(0, oldQ, reward, true);

        return rewardSum;
    }

    public void agentPlanning(int state, int action, int newState, float reward)
    {
        var oldQ = getQValues(state, action);
        var newQ = getQValues(newState);
        var maxQ = newQ.Max();

        QUpdate(maxQ, oldQ, reward, state, action);
    }

    private int policy(int state)
    {
        
        int action = 0;

        if (Random.Range(0f,1f) < epsilon) { // Exploring
            action = Random.Range(0, numActions);
        }
        else { // Exploiting
            var q_vals = getQValues(state);
            action = IndexMax(q_vals);
        }

        return action;
    }

    private int greedyPolicy(int state)
    {
        int action = 0;
        // Exploiting
        var q_vals = getQValues(state);
        action = IndexMax(q_vals);

        return action;
    }

    private int greedyPolicy(int state, List<float> targetQ)
    {
        int action;
        // Exploiting
        var q_vals = getQValues(state, targetQ);
        action = IndexMax(q_vals);
        return action;
    }

    private int IndexMax(List<float> values)
    {
        var maxVal = values.Max();
        var indexList = new List<int>();

        int i = 0;
        foreach (float value in values)
        {
            if (value == maxVal) { indexList.Add(i); }
            i++;
        }

        return indexList[Random.Range(0, indexList.Count)];
    }

    private List<float> getQValues(int state)
    {
        var valueList = new List<float>();

        // Get Q values of current State
        for (int i = 0; i < numActions; i++)
        {
            valueList.Add(q[state * numActions + i]);
        }

        return valueList;
    }

    private List<float> getQValues(int state, List<float> targetQ)
    {
        var valueList = new List<float>();
        // Get Q values of current State
        for (int i = 0; i < numActions; i++)
        {
            valueList.Add(targetQ[state * numActions + i]);
        }

        return valueList;
    }

    private float getQValues(int state, int action)
    {
        return q[state * numActions + action];
    }

    private void QUpdate(float maxQ, float oldQ, float reward, bool terminal)
    {
        if (!terminal)
        {
            q[lastState * numActions + lastAction] += alpha * (reward + gamma*maxQ - oldQ);
        }
        else
        {
            q[lastState * numActions + lastAction] += alpha * (reward - oldQ);
        } 
        
    }

    private void QUpdate(float maxQ, float oldQ, float reward, int state, int action)
    {
       q[state * numActions + action] += alpha * (reward + gamma * maxQ - oldQ);
    }

    public void printAllQValues()
    {
        foreach (float value in q)
        {
            Debug.Log(value);
        }
    }

    public List<float> GetQ()
    {
        return q;
    }

    public List<int> VisualizeGreedyPolicy(List<GameObject> arrowList, List<float> targetQ, bool shouldActivate=false)
    {
        var indexList = new List<int>();
        for (int i=0; i < numStates - 9; i++)
        {
            var index = greedyPolicy(i, targetQ);
            
            var rot = 0;

            switch (index)
            {
                case 0:
                    rot = 180;
                    break;
                case 1:
                    rot = 0;
                    break;
                case 2:
                    rot = -90;
                    break;
                case 3:
                    rot = 90;
                    break;
                case 4:
                    rot = -135;
                    break;
                case 5:
                    rot = -45;
                    break;
                case 6:
                    rot = 45;
                    break;
                case 7:
                    rot = 135;
                    break;
            }
            indexList.Add(index);

            // Rotate Arrow
            arrowList[i].transform.rotation = Quaternion.Euler(Vector3.up * rot);
            if (shouldActivate) { arrowList[i].SetActive(true); }
            
           
        }
        return indexList;
    }



}
