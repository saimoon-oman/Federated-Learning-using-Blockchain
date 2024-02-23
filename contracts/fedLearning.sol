// SPDX-License-Identifier: MIT
pragma solidity >=0.4.22 <0.9.0;
pragma experimental ABIEncoderV2;

contract FedLearning{
    string server;
    uint clientCount;
    mapping(address => string) clientWeights;

    mapping(address => uint) clientScore;

    uint accuracy;

    constructor() {
        // Initilizing default values
        clientCount = 0;
        accuracy = 0;
    }

    function setScore(address x, uint score) public {
        clientScore[x] = score;
    }

    function getScore(address x) public view returns(uint) {
        return clientScore[x];
    }

    function sendWeights(address x, string memory y) public {
        clientWeights[x] = y;
        clientCount += 1;
    }

    function setServer(string memory serverHash) public returns(bool){
        server = serverHash;
        clientCount = 0;
        return true;
    }


    function getServer() public view returns(string memory){
        return server;
        // return test;
    }

    function getWeights(address a) public view returns(string memory){
        // assert(msg.sender == a);
        return clientWeights[a];
    }
    

}