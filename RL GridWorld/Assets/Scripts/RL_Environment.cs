using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RL_Environment
{
    private Vector3 startPos;
    private Vector3 currentPos;

    private bool terminal;
    private int reward;

    public RL_Environment(Vector3 startPos)
    {
        this.startPos = startPos;
    }



}
