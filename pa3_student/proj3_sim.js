/*
 * Global variables
 */
var meshResolution;

// Particle states
var mass;
var vertexPosition, vertexNormal;
var vertexVelocity;

// Spring properties
var K, restLength; 

// Force parameters
var Cd;
var uf, Cv;


/*
 * Getters and setters
 */
function getPosition(i, j) {
    var id = i*meshResolution + j;
    return vec3.create([vertexPosition[3*id], vertexPosition[3*id + 1], vertexPosition[3*id + 2]]);
}

function setPosition(i, j, x) {
    var id = i*meshResolution + j;
    vertexPosition[3*id] = x[0]; vertexPosition[3*id + 1] = x[1]; vertexPosition[3*id + 2] = x[2];
}

function getNormal(i, j) {
    var id = i*meshResolution + j;
    return vec3.create([vertexNormal[3*id], vertexNormal[3*id + 1], vertexNormal[3*id + 2]]);
}

function getVelocity(i, j) {
    var id = i*meshResolution + j;
    return vec3.create(vertexVelocity[id]);
}

function setVelocity(i, j, v) {
    var id = i*meshResolution + j;
    vertexVelocity[id] = vec3.create(v);
}


/*
 * Provided global functions (you do NOT have to modify them)
 */
function computeNormals() {
    var dx = [1, 1, 0, -1, -1, 0], dy = [0, 1, 1, 0, -1, -1];
    var e1, e2;
    var i, j, k = 0, t;
    for ( i = 0; i < meshResolution; ++i )
        for ( j = 0; j < meshResolution; ++j ) {
            var p0 = getPosition(i, j), norms = [];
            for ( t = 0; t < 6; ++t ) {
                var i1 = i + dy[t], j1 = j + dx[t];
                var i2 = i + dy[(t + 1) % 6], j2 = j + dx[(t + 1) % 6];
                if ( i1 >= 0 && i1 < meshResolution && j1 >= 0 && j1 < meshResolution &&
                     i2 >= 0 && i2 < meshResolution && j2 >= 0 && j2 < meshResolution ) {
                    e1 = vec3.subtract(getPosition(i1, j1), p0);
                    e2 = vec3.subtract(getPosition(i2, j2), p0);
                    norms.push(vec3.normalize(vec3.cross(e1, e2)));
                }
            }
            e1 = vec3.create();
            for ( t = 0; t < norms.length; ++t ) vec3.add(e1, norms[t]);
            vec3.normalize(e1);
            vertexNormal[3*k] = e1[0];
            vertexNormal[3*k + 1] = e1[1];
            vertexNormal[3*k + 2] = e1[2];
            ++k;
        }
}


var clothIndex, clothWireIndex;
function initMesh() {
    var i, j, k;

    vertexPosition = new Array(meshResolution*meshResolution*3);
    vertexNormal = new Array(meshResolution*meshResolution*3);
    clothIndex = new Array((meshResolution - 1)*(meshResolution - 1)*6);
    clothWireIndex = [];

    vertexVelocity = new Array(meshResolution*meshResolution);
    restLength[0] = 4.0/(meshResolution - 1);
    restLength[1] = Math.sqrt(2.0)*4.0/(meshResolution - 1);
    restLength[2] = 2.0*restLength[0];

    for ( i = 0; i < meshResolution; ++i )
        for ( j = 0; j < meshResolution; ++j ) {
            setPosition(i, j, [-2.0 + 4.0*j/(meshResolution - 1), -2.0 + 4.0*i/(meshResolution - 1), 0.0]);
            setVelocity(i, j, vec3.create());

            if ( j < meshResolution - 1 )
                clothWireIndex.push(i*meshResolution + j, i*meshResolution + j + 1);
            if ( i < meshResolution - 1 )
                clothWireIndex.push(i*meshResolution + j, (i + 1)*meshResolution + j);
            if ( i < meshResolution - 1 && j < meshResolution - 1 )
                clothWireIndex.push(i*meshResolution + j, (i + 1)*meshResolution + j + 1);
        }
    computeNormals();

    k = 0;
    for ( i = 0; i < meshResolution - 1; ++i )
        for ( j = 0; j < meshResolution - 1; ++j ) {
            clothIndex[6*k] = i*meshResolution + j;
            clothIndex[6*k + 1] = i*meshResolution + j + 1;
            clothIndex[6*k + 2] = (i + 1)*meshResolution + j + 1;
            clothIndex[6*k + 3] = i*meshResolution + j;
            clothIndex[6*k + 4] = (i + 1)*meshResolution + j + 1;            
            clothIndex[6*k + 5] = (i + 1)*meshResolution + j;
            ++k;
        }
}


/*
 * KEY function: simulate one time-step using Euler's method
 */
function simulate(stepSize) {
    // FIX ME
    for ( i = 0; i < meshResolution; ++i )
        for ( j = 0; j < meshResolution; ++j ) {

            if (isFixedPoint(i, j))
                continue

            let forceSpring = calculateTotalSpringForce(i, j);
            let forceGravity = vec3.create([0, mass * -9.8 * 0.1, 0]);
            let forceDamping = calculateDampingForce(i, j);
            let forceViscous = calculateViscousForce(i, j);

            let forceNet = vec3.add(forceSpring, forceGravity);
            forceNet = vec3.add(forceNet, forceDamping);
            forceNet = vec3.add(forceNet, forceViscous);

            let currentVelocity = getVelocity(i, j);

            let nextVelocity = vec3.create([0, 0, 0]);
            nextVelocity = vec3.scale(vec3.create(forceNet), stepSize);
            nextVelocity = vec3.add(vec3.create(nextVelocity), currentVelocity);

            setVelocity(i, j, nextVelocity);

            let currentPosition = getPosition(i, j);
            let nextPosition = vec3.create([0, 0, 0])
            nextPosition = vec3.scale(vec3.create(nextVelocity), stepSize);
            nextPosition = vec3.add(vec3.create(currentPosition), nextPosition);

            setPosition(i, j, nextPosition);
        }
}

function isFixedPoint(i, j) {
    return (i === meshResolution - 1 && j === 0) || (i === meshResolution - 1 && j === meshResolution - 1);
}

function calculateTotalSpringForce(i, j) {
    let force = vec3.create([0, 0, 0]);

    // 1. structural spring
    let p = getPosition(i, j);

    let q = [];
    if (j < meshResolution - 1)
        q[q.length] = getPosition(i, j+1);

    if (i < meshResolution - 1)
        q[q.length] = getPosition(i+1, j);

    if (j > 0)
        q[q.length] = getPosition(i, j-1);

    if (i > 0)
        q[q.length] = getPosition(i-1, j);

    vec3.add(force, calculateSubSpringForce(0, p, q));

    // 2. shear spring
    p = getPosition(i, j);
    q = [];

    if (i < meshResolution - 1 && j < meshResolution - 1)
        q[q.length] = getPosition(i+1, j+1);

    if (i < meshResolution - 1 && j > 0)
        q[q.length] = getPosition(i+1, j-1);

    if (i > 0 && j < meshResolution - 1)
        q[q.length] = getPosition(i-1, j+1);

    if (i > 0 && j > 0)
        q[q.length] = getPosition(i-1, j-1);

    vec3.add(force, calculateSubSpringForce(1, p, q));

    // 3. flexion spring
    p = getPosition(i, j);
    q = [];

    if (j < meshResolution - 2)
        q[q.length] = getPosition(i, j+2);

    if (i < meshResolution - 2)
        q[q.length] = getPosition(i+2, j);

    if (j > 1)
        q[q.length] = getPosition(i, j-2);

    if (i > 1)
        q[q.length] = getPosition(i-2, j);

    vec3.add(force, calculateSubSpringForce(2, p, q));

    return force;
}

function calculateSubSpringForce(springIdx, p, q) {
    let force = vec3.create([0, 0, 0]);

    for (let i = 0; i < q.length; i++) {
        let pMinusQ = vec3.subtract(vec3.create(p), q[i]);
        let size = K[springIdx] * (restLength[springIdx] - vec3.length(pMinusQ));
        let direction = vec3.normalize(vec3.create(pMinusQ));

        vec3.add(force, vec3.scale(direction, size));
    }

    return force;
}

function calculateDampingForce(i, j) {
    let currentVelocity = getVelocity(i, j);
    return vec3.scale(currentVelocity, Cd);
}

function calculateViscousForce(i, j) {
    const normal = getNormal(i, j);
    let coefficient = Cv * vec3.dot(normal, vec3.subtract(vec3.create(uf), getVelocity(i, j)));
    return vec3.scale(normal, coefficient);
}