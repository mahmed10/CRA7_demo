"use strict";

// Space3DViewer is an extension of a Viewer that implements the common visualization
// framework for 3D stuff.
// Space3DViewer implements drawing functionality, but does not implement any
// message decoding functionality. Child classes that inherit from Space3DViewer
// should decode a message and instruct the plotting framework what to do.

class Space3DViewer extends Viewer {
  /**
    * Gets called when Viewer is first initialized.
    * @override
  **/
  onCreate() {
    // silly css nested div hell to force 100% width
    // but keep 1:1 aspect ratio

    this.wrapper = $('<div></div>')
      .css({
        "position": "relative",
        "width": "100%",
      })
      .appendTo(this.card.content);

    this.wrapper2 = $('<div></div>')
      .css({
        "width": "100%",
        "height": "0",
        "padding-bottom": "100%",
        "background": "#4e4e4c",
        "position": "relative",
        "overflow": "hidden",
      })
      .appendTo(this.wrapper);

    let that = this;

    this.gl = GL.create({ version:1, width: 500, height: 500});
    this.wrapper2[0].appendChild(this.gl.canvas);
    $(this.gl.canvas).css("width", "100%");
    this.gl.animate(); // launch loop

    this.cam_pos = [0,100,100];
    this.cam_theta = -1.5707;
    this.cam_phi = 1.0;
    this.cam_r = 50.0;
    this.cam_offset_x = 0.0;
    this.cam_offset_y = 0.0;
    this.cam_offset_z = 0.0;

    this.drawObjectsGl = null;

    //create basic matrices for cameras and transformation
    this.proj = mat4.create();
    this.view = mat4.create();
    this.model = mat4.create();
    this.mvp = mat4.create();
    this.temp = mat4.create();

    this.gl.captureMouse(true, true);
    this.gl.onmouse = function(e) {
      if(e.dragging) {
        if(e.rightButton) {
          that.cam_offset_x += e.deltax/30 * Math.sin(that.cam_theta);
          that.cam_offset_y -= e.deltax/30 * Math.cos(that.cam_theta);
          that.cam_offset_z += e.deltay/30;
          that.updatePerspective();
        } else {
          if(Math.abs(e.deltax) > 100 || Math.abs(e.deltay) > 100) return;
          that.cam_theta -= e.deltax / 300;
          that.cam_phi -= e.deltay / 300;

          // avoid euler singularities
          // also don't let the user flip the entire cloud around
          if(that.cam_phi < 0) {
            that.cam_phi = 0.001;
          }
          if(that.cam_phi > Math.PI) {
            that.cam_phi = Math.PI - 0.001;
          }
          that.updatePerspective();
        }
      }
    }

    this.gl.onmousewheel = function(e) {
      that.cam_r -= e.delta;
      if(that.cam_r < 1.0) that.cam_r = 1.0;
      if(that.cam_r > 1000.0) that.cam_r = 1000.0;
      that.updatePerspective();
    }

    this.updatePerspective = () => {
      that.cam_pos[0] = that.cam_offset_x + that.cam_r * Math.sin(that.cam_phi) * Math.cos(that.cam_theta);
      that.cam_pos[1] = that.cam_offset_y + that.cam_r * Math.sin(that.cam_phi) * Math.sin(that.cam_theta);
      that.cam_pos[2] = that.cam_offset_z + that.cam_r * Math.cos(that.cam_phi);

      that.view = mat4.create();
      mat4.perspective(that.proj, 45 * DEG2RAD, that.gl.canvas.width / that.gl.canvas.height, 0.1, 1000);
      mat4.lookAt(that.view, that.cam_pos, [this.cam_offset_x,this.cam_offset_y, this.cam_offset_z], [0,0,1]);
      mat4.multiply(that.mvp, that.proj, that.view);
    }

    this.updatePerspective();

    this.shader = new Shader('\
      precision highp float;\
      attribute vec3 a_vertex;\
      attribute vec4 a_color;\
      uniform mat4 u_mvp;\
      varying vec4 v_color;\
      void main() {\
          v_color = a_color;\
          gl_Position = u_mvp * vec4(a_vertex,1.0);\
          gl_PointSize = 1.5;\
      }\
      ', '\
      precision highp float;\
      uniform vec4 u_color;\
      varying vec4 v_color;\
      void main() {\
        gl_FragColor = u_color * v_color;\
      }\
    ');
    //generic gl flags and settings
    this.gl.clearColor(0.1,0.1,0.1,1);
    this.gl.enable(this.gl.DEPTH_TEST); // Enable depth test
    this.gl.depthFunc(this.gl.LEQUAL); // Near things obscure far things

    //rendering loop
    this.gl.ondraw = function() {
      that.gl.clear( that.gl.COLOR_BUFFER_BIT | that.gl.DEPTH_BUFFER_BIT );
      if(!that.drawObjectsGl) return;
      for(let i in that.drawObjectsGl) {
        if(that.drawObjectsGl[i].type === "points") {
          that.shader.uniforms({
            u_color: [1,1,1,1],
            u_mvp: that.mvp
          }).draw(that.drawObjectsGl[i].mesh, gl.POINTS);
        } else if(that.drawObjectsGl[i].type === "lines") {
          that.shader.uniforms({
            u_color: [1,1,1,1],
            u_mvp: that.mvp
          }).draw(that.drawObjectsGl[i].mesh, gl.LINES);
        } else if(that.drawObjectsGl[i].type === "triangles") {
          that.shader.uniforms({
            u_color: [1,1,1,1],
            u_mvp: that.mvp
          }).draw(that.drawObjectsGl[i].mesh, gl.TRIANGLES);
        }
      }
    };

    // initialize static mesh for grid

    this.gridPoints = [];
    this.gridColors = [];
    for(let x=-5.0;x<=5.0+0.001;x+=1.0) {
      this.gridPoints.push(x);
      this.gridPoints.push(-5);
      this.gridPoints.push(0);
      this.gridPoints.push(x);
      this.gridPoints.push(5);
      this.gridPoints.push(0);
      for(let i=0;i<8;i++) {
        this.gridColors.push(1);
      }
    }

    for(let y=-5.0;y<=5.0+0.001;y+=1.0) {
      this.gridPoints.push(-5);
      this.gridPoints.push(y);
      this.gridPoints.push(0);
      this.gridPoints.push(5);
      this.gridPoints.push(y);
      this.gridPoints.push(0);
      for(let i=0;i<8;i++) {
        this.gridColors.push(1);
      }
    }

    this.gridMesh = GL.Mesh.load({vertices: this.gridPoints, colors: this.gridColors}, null, null, this.gl);

    // initialize static mesh for toy car

    this.toyCarVertices = [
      // Car body (main part)
      -1.2, -0.5, -1.55, // bottom (scaled 30% less)
      1.2, -0.5, -1.55,
      1.2, 0.5, -1.55,
      -1.2, 0.5, -1.55,
      -1.2, -0.5, -0.85,
      1.2, -0.5, -0.85,
      1.2, 0.5, -0.85,
      -1.2, 0.5, -0.85,

      // Car roof (sloped part)
      -0.72, -0.4, -0.85, // bottom (scaled 30% less)
      0.72, -0.4, -0.85,
      0.72, 0.4, -0.85,
      -0.72, 0.4, -0.85,
      -0.72, -0.4, -0.36,
      0.72, -0.4, -0.36,
      0.72, 0.4, -0.36,
      -0.72, 0.4, -0.36,
    ].map(v => v / 3); // Scale the vertices to 1/3 size

    // Create vertical circular wheels
    this.createCircularWheel = (xOffset, yOffset, zOffset, radius, segments) => {
      let vertices = [];
      for (let i = 0; i < segments; i++) {
        let theta = 2 * Math.PI * i / segments;
        let nextTheta = 2 * Math.PI * (i + 1) / segments;
        vertices.push(xOffset, yOffset, zOffset);
        vertices.push(xOffset + radius * Math.cos(theta), yOffset, zOffset + radius * Math.sin(theta));
        vertices.push(xOffset + radius * Math.cos(nextTheta), yOffset, zOffset + radius * Math.sin(nextTheta));
      }
      return vertices;
    };

    let wheelSegments = 20;
    let wheelRadius = 0.2 / 3; // Scaled to 1/3 size
    this.toyCarVertices = this.toyCarVertices.concat(
      this.createCircularWheel(-0.96 / 3, -0.6 / 3, -1.69 / 3, wheelRadius, wheelSegments),
      this.createCircularWheel(0.96 / 3, -0.6 / 3, -1.69 / 3, wheelRadius, wheelSegments),
      this.createCircularWheel(-0.96 / 3, 0.6 / 3, -1.69 / 3, wheelRadius, wheelSegments),
      this.createCircularWheel(0.96 / 3, 0.6 / 3, -1.69 / 3, wheelRadius, wheelSegments)
    );

    this.toyCarColors = [
      // Car body
      1, 1, 0, 1,  // yellow
      1, 1, 0, 1,
      1, 1, 0, 1,
      1, 1, 0, 1,
      1, 1, 0, 1,
      1, 1, 0, 1,
      1, 1, 0, 1,
      1, 1, 0, 1,

      // Car roof
      0.83, 0.83, 0.83, 1,  // light grey
      0.83, 0.83, 0.83, 1,
      0.83, 0.83, 0.83, 1,
      0.83, 0.83, 0.83, 1,
      0.83, 0.83, 0.83, 1,
      0.83, 0.83, 0.83, 1,
      0.83, 0.83, 0.83, 1,
      0.83, 0.83, 0.83, 1,
    ];

    // Add black color for wheels
    let wheelColor = [];
    for (let i = 0; i < 4 * wheelSegments * 3; i++) {
      wheelColor.push(0, 0, 0, 1);
    }
    this.toyCarColors = this.toyCarColors.concat(wheelColor);

    this.toyCarIndices = [
      // Car body (main part)
      0, 1, 2,  0, 2, 3,
      4, 5, 6,  4, 6, 7,
      0, 1, 5,  0, 5, 4,
      2, 3, 7,  2, 7, 6,
      0, 3, 7,  0, 7, 4,
      1, 2, 6,  1, 6, 5,

      // Car roof (sloped part)
      8, 9, 10,  8, 10, 11,
      12, 13, 14,  12, 14, 15,
      8, 9, 13,  8, 13, 12,
      10, 11, 15,  10, 15, 14,
      8, 11, 15,  8, 15, 12,
      9, 10, 14,  9, 14, 13,
    ];

    // Add indices for circular wheels
    for (let i = 0; i < 4; i++) {
      let baseIndex = 16 + i * wheelSegments * 3;
      for (let j = 0; j < wheelSegments * 3; j += 3) {
        this.toyCarIndices.push(baseIndex + j, baseIndex + j + 1, baseIndex + j + 2);
      }
    }

    this.toyCarMesh = GL.Mesh.load({vertices: this.toyCarVertices, colors: this.toyCarColors, triangles: this.toyCarIndices}, null, null, this.gl);
  }

  _getColor(v, vmin, vmax) {
    // cube edge walk from from http://paulbourke.net/miscellaneous/colourspace/
    let c = [1.0, 1.0, 1.0];

    if (v < vmin)
       v = vmin;
    if (v > vmax)
       v = vmax;
    let dv = vmax - vmin;
    if(dv < 1e-2) dv = 1e-2;

    if (v == 0.0) {
      c[0] = 1;
    } else if (v == 170.0) {
      c[0] = 0;
      c[1] = 0;
    } else if (v < (vmin + 0.25 * dv)) {
      c[0] = 0;
      c[1] = 4 * (v - vmin) / dv;
    } else if (v < (vmin + 0.5 * dv)) {
      c[0] = 0;
      c[2] = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
    } else if (v < (vmin + 0.75 * dv)) {
      c[0] = 4 * (v - vmin - 0.5 * dv) / dv;
      c[2] = 0;
    } else {
      c[1] = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
      c[2] = 0;
    }

    return(c);
  }

  draw(drawObjects) {
    this.drawObjects = drawObjects;
    let drawObjectsGl = [];

    // draw grid
    
    // drawObjectsGl.push({type: "lines", mesh: this.gridMesh});

    // draw toy car

    drawObjectsGl.push({type: "triangles", mesh: this.toyCarMesh});

    for(let i in drawObjects) {
      let drawObject = drawObjects[i];
      if(drawObject.type === "points") {
        let colors = new Float32Array(drawObject.data.length / 3 * 4);
        let zmin = 0 || -2;
        let zmax = 255 || 2;
        let zrange = zmax - zmin;
        for(let j=0; j < drawObject.data.length / 3; j++) {
          let c = this._getColor(drawObject.intensities[j+2], zmin, zmax)
          colors[4*j] = c[0];
          colors[4*j+1] = c[1];
          colors[4*j+2] = c[2];
          colors[4*j+3] = 1;
        }
        let points = drawObject.data;
        drawObjectsGl.push({type: "points", mesh: GL.Mesh.load({vertices: points, colors: colors}, null, null, this.gl)});
      }
    }
    this.drawObjectsGl = drawObjectsGl;
  }
}

Space3DViewer.supportedTypes = [
];

Space3DViewer.maxUpdateRate = 10.0;

Viewer.registerViewer(Space3DViewer);
