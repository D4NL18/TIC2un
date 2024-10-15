import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class DeeplearningService {

  private apiUrl = 'http://localhost:5000';

  constructor(private http: HttpClient) { }

  trainDeepLearning(): Observable<any> {
    return this.http.post(`${this.apiUrl}/confusion-matrix/pt`, {});
  }

  getImagePT(type: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/confusion-matrix/${type}`, { responseType: 'blob' });
  }

  getImageTF(type: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/confusion-matrix/${type}`, { responseType: 'blob' });
  }

  getAccuracyPT(type: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/accuracy/${type}`);
  }
  getAccuracyTF(type: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/accuracy/${type}`);
  }
}